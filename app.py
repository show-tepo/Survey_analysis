import streamlit as st
import pandas as pd
import networkx as nx
import re
from itertools import combinations
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from janome.tokenizer import Tokenizer

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import chardet
import io

st.set_page_config(layout="wide")
st.title("アンケート分析ツール")
st.markdown("---")

DEFAULT_SETTINGS = {
    "stopwords": "こと もの よう ある する いる",
    "selected_pos": ["名詞", "動詞", "形容詞"],
    "layout_option": "spring_layout（自然な配置）",
    "spring_k": 0.1,
    "ngram": 1,
    "weight_threshold": 4,
    "node_size_multiplier": 0.2,
    "cooccurrence_unit": "セル単位",  # 文単位 or セル単位
}

def reset_to_defaults():
    for key, value in DEFAULT_SETTINGS.items():
        st.session_state[key] = value

# セッション初期化
for key, value in DEFAULT_SETTINGS.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header("パラメータ設定")

    # CSVファイル
    uploaded_file = st.file_uploader(
        "CSVファイルをアップロードしてください", 
        type=["csv"],
        help="解析したいCSVファイルを指定します。（例：アンケート結果など）"
    )
    if uploaded_file is not None:
        raw_data = uploaded_file.getvalue()
        result = chardet.detect(raw_data)
        detected_encoding = result["encoding"] or "utf-8"

        try:
            text_data = raw_data.decode(detected_encoding, errors="replace")
            df = pd.read_csv(io.StringIO(text_data))
        except Exception as e:
            st.error(f"CSV の読み込みに失敗しました。エンコード: {detected_encoding}\n\nError: {e}")
            df = None

        if df is not None:
            # 解析対象の列
            column = st.selectbox(
                "解析対象の列を選んでください",
                df.columns,
                key="target_column",
                help="CSVファイルに含まれる列（カラム）の中から、形態素解析の対象となる文章が書かれている列を選択します。"
            )

            # ストップワード
            st.text_area(
                "ストップワードをスペース区切りで入力",
                st.session_state["stopwords"],
                key="stopwords",
                help="解析から除外したい単語をスペース区切りで指定します。（例：する ある いる）"
            )

            # 品詞選択
            all_pos_options = [
                "名詞", "動詞", "形容詞", "副詞", "連体詞",
                "接続詞", "感動詞", "助詞", "助動詞", "記号"
            ]
            selected_pos = st.multiselect(
                "解析対象とする品詞",
                all_pos_options,
                default=st.session_state["selected_pos"],
                key="selected_pos",
                help="形態素解析で取り出したい品詞を選択します。（名詞、動詞、形容詞など）"
            )

            # 共起カウント単位（セル or 文）
            st.session_state["cooccurrence_unit"] = st.radio(
                "共起カウントの単位を選択",
                ["セル単位", "文単位"],
                index=0 if st.session_state["cooccurrence_unit"] == "セル単位" else 1,
                help=(
                    "【セル単位】1行のテキストを丸ごと1つの単位として扱い、行内で出現する単語を共起とみなします。\n"
                    "【文単位】1行を文ごとに区切り、同じ文の中に出現した単語だけを共起とみなします。"
                )
            )

            # ノード配置方法
            layout_option = st.selectbox(
                "ノード配置方法（レイアウト）",
                [
                    "spring_layout（自然な配置）",
                    "circular_layout（円形配置）",
                    "random_layout（ランダム配置）",
                ],
                index=[
                    "spring_layout（自然な配置）",
                    "circular_layout（円形配置）",
                    "random_layout（ランダム配置）",
                ].index(st.session_state["layout_option"]),
                key="layout_option",
                help="共起ネットワークをどのように配置するかを選べます。spring_layoutは自然に散らした配置になります。"
            )

            # spring_k (spring_layout 用パラメータ)
            if layout_option.startswith("spring"):
                spring_k = st.slider(
                    "自然長（spring k）",
                    0.01, 3.0,
                    st.session_state["spring_k"],
                    step=0.01,
                    key="spring_k",
                    help="spring_layoutのノード間のばねの自然長（k値）。値が小さいほどノードは密集し、大きいほど広がります。"
                )
            else:
                spring_k = None

            # n-gram
            ngram = st.slider(
                "n-gram（語数の単位）",
                1, 5,
                st.session_state["ngram"],
                key="ngram",
                help="形態素解析後、何語をまとめて1単位にするか指定します。1なら単語単位、2なら2-gramなど。"
            )

            # weight_threshold
            weight_threshold = st.slider(
                "共起回数の閾値",
                1, 10,
                st.session_state["weight_threshold"],
                key="weight_threshold",
                help="ネットワークを作る際、同じペアが何回以上共起したらエッジを作るかの閾値。"
            )

            # node_size_multiplier
            node_size_multiplier = st.slider(
                "ノードサイズの倍率",
                0.1, 5.0,
                st.session_state["node_size_multiplier"],
                step=0.1,
                key="node_size_multiplier",
                help="共起ネットワークにおけるノードサイズのスケールを調整します。"
            )

            # デフォルト設定に戻す
            if st.button("デフォルト設定に戻す", on_click=reset_to_defaults,
                         help="すべてのパラメータを初期状態に戻します。"):
                st.success("パラメータがデフォルト設定に戻りました！")

    else:
        df = None
        column = None

# ========================================================================================
# 以下、本体の解析ロジック・表示部分（セル単位と文単位の両方に対応したサンプル）
# ========================================================================================
if 'df' in locals() and df is not None and 'column' in locals() and column is not None:
    import re

    def split_into_sentences(text):
        """文単位で分割するための簡易関数。"""
        text = text.replace("\n", " ")
        # 句読点や感嘆符などで分割
        sentences = re.split(r'[。！？!?]', text)
        return [s.strip() for s in sentences if s.strip()]

    def preprocess_data(df, column, stopwords_str, selected_pos, ngram, cooccurrence_unit):
        tokenizer = Tokenizer()
        stopwords_list = stopwords_str.split()

        def extract_keywords(text):
            if not isinstance(text, str):
                return []
            words = []
            for token in tokenizer.tokenize(text):
                if any(token.part_of_speech.startswith(pos) for pos in selected_pos) and token.surface not in stopwords_list:
                    words.append(token.surface)
            # n-gram
            return [" ".join(words[i : i + ngram]) for i in range(len(words) - ngram + 1)]

        all_rows_keywords = []

        if cooccurrence_unit == "文単位":
            # 文ごとに分割してキーワード抽出
            for _, row in df.iterrows():
                text = row[column]
                sentences = split_into_sentences(text)
                keywords_by_sentence = [extract_keywords(s) for s in sentences]
                # rowごとに [[文1のkw],[文2のkw],...] の構造
                all_rows_keywords.append(keywords_by_sentence)
        else:
            # セル（行）単位
            for _, row in df.iterrows():
                text = row[column]
                row_kw = extract_keywords(text)
                # 統一感のため2次元リストにする
                all_rows_keywords.append([row_kw])

        df["keywords"] = all_rows_keywords

        # 出現頻度（word_count）はすべての文/行をまとめて数える
        all_keywords = []
        for row_kw_2d in all_rows_keywords:  # 2次元 [[kw_of_sentence], [kw_of_sentence2]...]
            for kw_list in row_kw_2d:
                all_keywords.extend(kw_list)
        word_count = Counter(all_keywords)

        return df, word_count

    df, word_count = preprocess_data(
        df,
        column,
        st.session_state["stopwords"],
        st.session_state["selected_pos"],
        st.session_state["ngram"],
        st.session_state["cooccurrence_unit"]
    )

    def create_frequency_chart(word_count):
        word_df = (
            pd.DataFrame(word_count.items(), columns=["単語", "出現回数"])
            .sort_values(by="出現回数", ascending=False)
        )
        return word_df

    def create_wordcloud(word_count, font_path="./font/SawarabiGothic-Regular.ttf"):
        wc = WordCloud(width=600, height=450, background_color="white", font_path=font_path)
        wc_img = wc.generate_from_frequencies(word_count)
        return wc_img.to_image()

    def create_cooccurrence_network(df, word_count, weight_threshold, node_size_multiplier, layout_option):
        cooccurrence_pairs = []
        for row_kw_2d in df["keywords"]:  # row_kw_2d = [[文1のkw], [文2のkw], ...]
            for kw_list in row_kw_2d:
                filtered_keywords = [w for w in kw_list if w in word_count]
                cooccurrence_pairs.extend(combinations(filtered_keywords, 2))

        cooccurrence_count = Counter(cooccurrence_pairs)
        G = nx.Graph()
        for (w1, w2), weight in cooccurrence_count.items():
            if weight >= weight_threshold:
                G.add_edge(w1, w2, weight=weight)

        if G.number_of_nodes() == 0:
            return go.Figure().update_layout(title="共起ネットワーク (エッジなし)", width=1200, height=800)

        if layout_option.startswith("spring"):
            pos = nx.spring_layout(G, seed=42, k=st.session_state["spring_k"], iterations=50)
        elif layout_option.startswith("circular"):
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)

        x_nodes = [pos[node][0] for node in G.nodes()]
        y_nodes = [pos[node][1] for node in G.nodes()]
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_sizes = [G.degree(node, weight="weight") * node_size_multiplier for node in G.nodes()]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines"
        )
        node_trace = go.Scatter(
            x=x_nodes, y=y_nodes,
            mode="markers+text",
            marker=dict(size=node_sizes, color="skyblue", line_width=2),
            text=list(G.nodes()),
            textposition="top center",
            hoverinfo="text",
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="共起ネットワーク",
            titlefont_size=16,
            showlegend=False,
            width=1200,
            height=800,
        )
        return fig

    # ===== 出現頻度分析 =====
    st.subheader("出現頻度分析")
    word_df = create_frequency_chart(word_count)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("単語出現頻度表（上位10件）")
        st.dataframe(word_df.head(10).reset_index(drop=True), use_container_width=True)

    with col2:
        bar_fig = px.bar(
            word_df.head(10),
            x="単語",
            y="出現回数",
            title="単語出現頻度（上位10件）"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # ===== ワードクラウド =====
    st.subheader("ワードクラウド")
    wc_image = create_wordcloud(dict(word_count))
    st.image(wc_image, use_container_width=True)

    # ===== 共起ネットワーク =====
    st.subheader("共起ネットワーク")
    fig = create_cooccurrence_network(
        df, word_count,
        st.session_state["weight_threshold"],
        st.session_state["node_size_multiplier"],
        st.session_state["layout_option"]
    )
    st.plotly_chart(fig, use_container_width=False)
