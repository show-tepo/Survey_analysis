import streamlit as st
import pandas as pd
import networkx as nx
from itertools import combinations
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from janome.tokenizer import Tokenizer

# ページ全体を広げて使用
st.set_page_config(layout="wide")

# アプリのタイトル
st.title("アンケート分析ツール")
st.markdown("---")

# デフォルト設定の辞書
DEFAULT_SETTINGS = {
    "stopwords": "こと もの よう ある する いる",
    "selected_pos": ["名詞", "動詞", "形容詞"],
    "layout_option": "spring_layout（自然な配置）",
    "spring_k": 0.1,
    "ngram": 1,
    "weight_threshold": 2,
    "node_size_multiplier": 1.0,
}

# デフォルト設定に戻す関数
def reset_to_defaults():
    for key, value in DEFAULT_SETTINGS.items():
        st.session_state[key] = value

# セッション状態の初期化
for key, value in DEFAULT_SETTINGS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# サイドバーでのパラメータ調整
with st.sidebar:
    st.header("パラメータ設定")

    # ファイルアップロード
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

    if uploaded_file is not None:
        # ファイルがアップロードされた場合に表示
        df = pd.read_csv(uploaded_file)

        # 解析対象の列を選択
        column = st.selectbox(
            "解析対象の列を選んでください",
            df.columns,
            key="target_column",
        )

        # ストップワードの入力（デフォルト値付き）
        stopwords_input = st.text_area(
            "ストップワードをスペース区切りで入力",
            st.session_state["stopwords"],
            key="stopwords",
        )

        # 品詞選択
        all_pos_options = [
            "名詞", "動詞", "形容詞", "副詞", "連体詞", "接続詞", "感動詞", "助詞", "助動詞", "記号"
        ]
        selected_pos = st.multiselect(
            "解析対象とする品詞を選んでください（複数選択可）",
            all_pos_options,
            default=st.session_state["selected_pos"],
            key="selected_pos",
        )

        # ノード配置方法（レイアウト）
        layout_option = st.selectbox(
            "ノード配置方法（レイアウト）を選択",
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
        )

        # spring_layout用パラメータ
        if layout_option.startswith("spring"):
            spring_k = st.slider(
                "自然長（spring k）",
                0.01,
                3.0,
                st.session_state["spring_k"],
                step=0.01,
                key="spring_k",
            )
        else:
            spring_k = None

        # n-gramの選択
        ngram = st.slider(
            "n-gram（語数の単位）",
            1,
            5,
            st.session_state["ngram"],
            key="ngram",
        )

        # 共起ネットワーク作成パラメータ
        weight_threshold = st.slider(
            "共起回数の閾値（エッジを作成する最小回数）",
            1,
            10,
            st.session_state["weight_threshold"],
            key="weight_threshold",
        )
        node_size_multiplier = st.slider(
            "ノードサイズの倍率",
            0.1,
            5.0,
            st.session_state["node_size_multiplier"],
            step=0.1,
            key="node_size_multiplier",
        )

        # デフォルト設定に戻すボタン（サイドバーの一番下に配置）
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("デフォルト設定に戻す", on_click=reset_to_defaults):
            st.success("パラメータがデフォルト設定に戻りました！")

if uploaded_file is not None:
    # 共通データ処理関数
    def preprocess_data(df, column, stopwords, selected_pos, ngram):
        tokenizer = Tokenizer()

        # 指定された品詞を抽出
        def extract_keywords(text):
            words = []
            for token in tokenizer.tokenize(text):
                if any(token.part_of_speech.startswith(pos) for pos in selected_pos) and token.surface not in stopwords:
                    words.append(token.surface)
            # n-gramを生成
            return [" ".join(words[i:i + ngram]) for i in range(len(words) - ngram + 1)]

        # キーワードの抽出
        df["keywords"] = df[column].apply(extract_keywords)
        all_keywords = [keyword for keywords in df["keywords"] for keyword in keywords]

        # 単語頻度の計算
        word_count = Counter(all_keywords)
        return df, word_count

    # データを前処理
    df, word_count = preprocess_data(df, column, stopwords_input.split(), selected_pos, ngram)

    # 出現頻度分析の表とグラフを作成
    def create_frequency_chart(word_count):
        word_df = pd.DataFrame(word_count.items(), columns=["単語", "出現回数"]).sort_values(by="出現回数", ascending=False)
        return word_df

    # 共起ネットワーク作成関数
    def create_cooccurrence_network(df, word_count, weight_threshold, node_size_multiplier, layout_option):
        # 共起ペアを作成
        cooccurrence_pairs = []
        for keywords in df["keywords"]:
            filtered_keywords = [word for word in keywords if word in word_count]
            cooccurrence_pairs.extend(combinations(filtered_keywords, 2))
        cooccurrence_count = Counter(cooccurrence_pairs)

        # ネットワークの作成
        G = nx.Graph()
        for (word1, word2), weight in cooccurrence_count.items():
            # 閾値を適用
            if weight >= weight_threshold:
                G.add_edge(word1, word2, weight=weight)

        # ノードが存在しない場合は空のグラフを返す
        if G.number_of_nodes() == 0:
            return go.Figure().update_layout(title="共起ネットワーク (エッジなし)", width=1200, height=800)

        # レイアウトの選択
        if layout_option.startswith("spring"):
            pos = nx.spring_layout(G, seed=42, k=st.session_state["spring_k"], iterations=50)
        elif layout_option.startswith("circular"):
            pos = nx.circular_layout(G)
        elif layout_option.startswith("random"):
            pos = nx.random_layout(G)

        # ノードとエッジの座標を計算
        x_nodes = [pos[node][0] for node in G.nodes()]
        y_nodes = [pos[node][1] for node in G.nodes()]
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # ノードサイズを計算
        node_sizes = [G.degree(node, weight="weight") * node_size_multiplier for node in G.nodes()]

        # エッジトレース
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines"
        )

        # ノードトレース
        node_trace = go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers+text",
            marker=dict(size=node_sizes, color="skyblue", line_width=2),
            text=list(G.nodes()),
            textposition="top center",
            hoverinfo="text",
        )

        # グラフサイズを調整
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="共起ネットワーク",
            titlefont_size=16,
            showlegend=False,
            width=1200,
            height=800,
        )
        return fig


    # 出現頻度分析
    st.subheader("出現頻度分析")
    word_df = create_frequency_chart(word_count)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("単語出現頻度表")
        st.dataframe(word_df.head(10).reset_index(drop=True), use_container_width=True)

    with col2:
        bar_fig = px.bar(word_df.head(10), x="単語", y="出現回数", title="単語出現頻度（上位10件）")
        st.plotly_chart(bar_fig, use_container_width=True)

    # 共起ネットワーク
    st.subheader("共起ネットワーク分析")
    fig = create_cooccurrence_network(df, word_count, weight_threshold, node_size_multiplier, layout_option)
    st.plotly_chart(fig, use_container_width=True)
