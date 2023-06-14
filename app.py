

import pandas as pd
import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import normalize
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import chi2_contingency


directory = '/media/jahaziel/Datos/proyectos/Smarketing/dataset/Google_Reviews/Donosti/2023-06-14/'

# print(__file__)

# directory = __file__.split('app.py')[0]

st.set_page_config(page_title='Prueba Google Reviews', layout='wide')


WIDTH = 1000
HEIGHT = 600
FACTOR = 5 # 10% de los datos

st.title(f'''Google Reviews''')

text_df = pd.read_csv(f'''{directory}/text_info_sample.csv''')
best_text_by_topics_df = pd.read_csv(f'''{directory}/best_text_by_topics.csv''')
topic_word_weights_df = pd.read_csv(f'''{directory}/topic_word_weights.csv''')
topic_labels_df = pd.read_csv(f'''{directory}/topic_labels.csv''')
topics_over_time_df = pd.read_csv(f'''{directory}/topics_over_time.csv''')
bertopic_embeddings_df = pd.read_csv(f'''{directory}/topic_embeddings.csv''')
bertopic_embeddings = bertopic_embeddings_df.values



if 'sentence_object' not in st.session_state:
    st.session_state['sentence_object'] = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

if 'general_topic_plot' not in st.session_state:
    st.session_state['general_topic_plot'] = None

if 'general_place_plot' not in st.session_state:
    st.session_state['general_place_plot'] = None


text_df['date'] = pd.to_datetime(text_df['date']).dt.date

topics_over_time_df = topics_over_time_df[topics_over_time_df['Topic'] != -1]
topics_over_time_df['Timestamp'] = pd.to_datetime(topics_over_time_df['Timestamp']).dt.date
topics_over_time_df = topics_over_time_df.reset_index(drop=True)

dict_topic_labels = dict(zip(topic_labels_df['Topic'], topic_labels_df['Name']))


names = list(set(text_df['name']))
place_id_new_names = {}
for name in names:
    _df = text_df[text_df['name'] == name]
    place_ids = list(set(_df['place_id']))
    if len(place_ids) == 1:
        place_id_new_names[place_ids[0]] = name
    else:
        for enum, place_id in enumerate(place_ids):
            place_id_new_names[place_id] = f'''{name} ({enum})'''


text_df['place_id'] = text_df['place_id'].map(place_id_new_names)
best_text_by_topics_df['place_id'] = best_text_by_topics_df['place_id'].map(place_id_new_names)


text_df['stars'].fillna(0, inplace=True)

# ====================================================================================================

tab1, tab2, tab3, tab4 = st.tabs(['GENERAL', 'PLACE_ID', 'Buscador', 'Comparador'])

with tab1:
    st.markdown(f'''## Estadísticas Generales''')
    st.markdown(
        f'''
        * Número de textos publicados: {text_df.shape[0] * FACTOR}
        * Sentimiento negativo: {text_df['negative_score'].mean().round(2) * 100}%
        * Sentimiento neutral: {text_df['neutral_score'].mean().round(2) * 100}%
        * Sentimiento positivo: {text_df['positive_score'].mean().round(2) * 100}%
        ''')

    st.markdown(f'''## Estadísticas a lo largo del tiempo''')
    statistics_date_df = (text_df.groupby('date').size() * FACTOR).reset_index(name='counts')

    fig = make_subplots()
    fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['counts'], name='Número de textos publicados'), secondary_y=False)
    fig.update_layout(title='Número de textos publicados a lo largo del tiempo', width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'''## Principales Place_ID ''')
    place_id_df = text_df.groupby('place_id').size().reset_index(name='counts')
    place_id_df = place_id_df.sort_values(by='counts', ascending=False).reset_index(drop=True)
    place_id_df = place_id_df.head(20)
    fig = make_subplots()
    fig.add_trace(go.Bar(x=place_id_df['place_id'], y=place_id_df['counts'], name='Número de textos publicados'), secondary_y=False)
    fig.update_layout(title='Principales Place_ID', width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig, use_container_width=True)

    tab_1_1, tab_1_2 = st.tabs(['Topicos', 'PLACE_ID'])
    with tab_1_1:
        if st.session_state['general_topic_plot'] is None:
            fig = go.Figure()
            _df = text_df[text_df['topic'] == -1]
            fig.add_trace(go.Scatter(x=_df['x'], y=_df['y'], hovertext=_df['review'], hoverinfo='text', mode='markers+text', name='Sin tópico', marker=dict(color='#CFD8DC', size=5, opacity=0.5), showlegend=False))
            all_topics = sorted(text_df['topic'].unique())
            for topic in all_topics:
                if int(topic) == -1:
                    continue
                selection = text_df[text_df['topic'] == topic]
                label_name = dict_topic_labels[topic]
                fig.add_trace(go.Scatter(x=selection['x'], y=selection['y'], hovertext=selection['review'], hoverinfo='text', mode='markers+text', name=label_name, marker=dict(size=5, opacity=0.5)))
            x_range = [text_df['x'].min() - abs(text_df['x'].min() * 0.15), text_df['x'].max() + abs(text_df['x'].max() * 0.15)]
            y_range = [text_df['y'].min() - abs(text_df['y'].min() * 0.15), text_df['y'].max() + abs(text_df['y'].max() * 0.15)]
            fig.add_shape(type="rect", x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1], line=dict(color="#CFD8DC", width=2))
            fig.add_shape(type="rect", x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2, line=dict(color="#CFD8DC", width=2))
            fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
            fig.add_annotation(x=sum(x_range) / 2, y=y_range[1], text="D2", showarrow=False, xshift=10)
            fig.update_layout(template='simple_white', title={'text': "<b>", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22, color='Black')})
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(width=WIDTH * 1.5, height=HEIGHT * 1.5)
            st.session_state['general_topic_plot'] = fig
        st.plotly_chart(st.session_state['general_topic_plot'], use_container_width=True)
    with tab_1_2:
        if st.session_state['general_place_plot'] is None:
            fig = go.Figure()
            all_places = place_id_df['place_id'].unique()
            for place_id in all_places:
                selection = text_df[text_df['place_id'] == place_id]
                fig.add_trace(go.Scatter(x=selection['x'], y=selection['y'], hovertext=selection['review'], hoverinfo='text', mode='markers+text', name=place_id, marker=dict(size=5, opacity=0.5)))
            x_range = [text_df['x'].min() - abs(text_df['x'].min() * 0.15), text_df['x'].max() + abs(text_df['x'].max() * 0.15)]
            y_range = [text_df['y'].min() - abs(text_df['y'].min() * 0.15), text_df['y'].max() + abs(text_df['y'].max() * 0.15)]
            fig.add_shape(type="rect", x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1], line=dict(color="#CFD8DC", width=2))
            fig.add_shape(type="rect", x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2, line=dict(color="#CFD8DC", width=2))
            fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
            fig.add_annotation(x=sum(x_range) / 2, y=y_range[1], text="D2", showarrow=False, xshift=10)
            fig.update_layout(template='simple_white', title={'text': "<b>", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22, color='Black')})
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(width=WIDTH * 1.5, height=HEIGHT * 1.5)
            st.session_state['general_place_plot'] = fig
        st.plotly_chart(st.session_state['general_place_plot'], use_container_width=True)


    st.markdown(f'''## Evolución de tópicos ''')
    normalize_frequency = False
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]
    general_topics_over_time_df = topics_over_time_df[topics_over_time_df['place_id'] == 'general'].reset_index(drop=True)
    general_topics_over_time_df = general_topics_over_time_df.sort_values(["Topic", "Timestamp"])
    general_topics_over_time_df['Timestamp'] = pd.to_datetime(general_topics_over_time_df['Timestamp'])
    general_topics_over_time_df["Name"] = general_topics_over_time_df.Topic.apply(lambda x: dict_topic_labels[x])
    fig = go.Figure()
    for index, topic in enumerate(general_topics_over_time_df.Topic.unique()):
        trace_data = general_topics_over_time_df.loc[general_topics_over_time_df.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y, mode='lines+markers', marker_color=colors[index % 7], hoverinfo="text", name=topic_name, hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout( yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency", template="simple_white", hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))
    fig.update_layout(width=WIDTH * 1.5, height=HEIGHT)
    st.plotly_chart(fig)


with tab2:
    place_id = st.selectbox('Elige Place_id', text_df['place_id'].unique().tolist())
    place_text_df = text_df[text_df['place_id'] == place_id]
    place_topics_over_time_df = topics_over_time_df[topics_over_time_df['place_id'] == place_id].reset_index(drop=True)
    topic_count_sentiment_df = place_text_df.groupby('topic').agg({'id': 'count', 'negative_score': 'mean', 'neutral_score': 'mean', 'positive_score': 'mean'}).reset_index()

    st.markdown(f'''## {place_id}''')

    st.markdown(f'''### Estadísticas Generales''')
    st.markdown(
        f'''
        * Número de textos publicados: {place_topics_over_time_df['Frequency'].sum()}
        * Sentimiento negativo: {place_text_df['negative_score'].mean().round(2) * 100}%
        * Sentimiento neutral: {place_text_df['neutral_score'].mean().round(2) * 100}%
        * Sentimiento positivo: {place_text_df['positive_score'].mean().round(2) * 100}%
        ''')

    statistics_date_df = (place_topics_over_time_df.groupby('Timestamp')['Frequency'].sum()).reset_index()

    st.markdown(f'''### Evolución de textos, ranking y  sentimiento ''')

    tab_2_1, tab_2_2 = st.tabs(['Sentimiento', 'Ranking'])
    with tab_2_1:
        sentiment_date_df = place_text_df.groupby('date')[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=statistics_date_df['Timestamp'], y=statistics_date_df['Frequency'], name='Número de textos publicados'), secondary_y=True)
        fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
        fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
        fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
        fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Sentimiento de los textos a lo largo del tiempo')
        fig.update_layout(width=WIDTH, height=HEIGHT)
        st.plotly_chart(fig, use_container_width=True)
    with tab_2_2:
        ranking_date_df = place_text_df.groupby('date')[['stars']].mean().reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=statistics_date_df['Timestamp'], y=statistics_date_df['Frequency'], name='Número de textos publicados'), secondary_y=True)
        fig.add_trace(go.Scatter(x=ranking_date_df['date'], y=ranking_date_df['stars'], name='Ranking'), secondary_y=False)
        fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos por tópico')
        fig.update_layout(width=WIDTH, height=HEIGHT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'''### Cantidad, raking y sentimiento por tópicos ''')
    tab_2_3, tab_2_4 = st.tabs(['Cantidad', 'Ranking'])
    with tab_2_3:
        topic_count_sentiment_df['id'] = topic_count_sentiment_df['id'] * FACTOR
        topic_count_sentiment_df = topic_count_sentiment_df[topic_count_sentiment_df['topic'] != -1].sort_values('topic')
        topic_count_sentiment_df['topic_label'] = [dict_topic_labels[x] for x in topic_count_sentiment_df['topic']]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['id'], name='Cantidad de textos', mode='lines+markers'), secondary_y=True)
        fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
        fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
        fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
        fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos por tópico')
        fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)
        st.plotly_chart(fig, use_container_width=True)

    with tab_2_4:
        ranking_date_df = place_text_df.groupby('topic')[['stars']].mean().reset_index()
        ranking_date_df['topic_label'] = [dict_topic_labels[x] for x in ranking_date_df['topic']]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['id'], name='Cantidad de textos', mode='lines+markers'), secondary_y=True)
        fig.add_trace(go.Bar(x=ranking_date_df['topic_label'], y=ranking_date_df['stars'], name='Ranking'), secondary_y=False)
        fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos por tópico')
        fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'''### Evolución de tópicos ''')
    normalize_frequency = False
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]
    place_topics_over_time_df = topics_over_time_df[topics_over_time_df['place_id'] == place_id].reset_index(drop=True)
    place_topics_over_time_df = place_topics_over_time_df.sort_values(["Topic", "Timestamp"])
    place_topics_over_time_df['Timestamp'] = pd.to_datetime(place_topics_over_time_df['Timestamp'])
    place_topics_over_time_df["Name"] = place_topics_over_time_df.Topic.apply(lambda x: dict_topic_labels[x])
    fig = go.Figure()
    for index, topic in enumerate(place_topics_over_time_df.Topic.unique()):
        trace_data = place_topics_over_time_df.loc[place_topics_over_time_df.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y, mode='lines+markers', marker_color=colors[index % 7],
                                 hoverinfo="text", name=topic_name,
                                 hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
                      template="simple_white",
                      hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))
    fig.update_layout(width=WIDTH * 1.5, height=HEIGHT)
    st.plotly_chart(fig)

    st.markdown(f'''### Principales comentarios por tópico ''')
    order_topics = topic_count_sentiment_df.sort_values('id', ascending=False)['topic'].values[:5]
    for topic in order_topics:
        topic_label = dict_topic_labels[topic]
        st.markdown(f'''#### {topic_label}''')
        topic_df = place_text_df[place_text_df['topic'] == topic].head(5)
        topic_df = topic_df.sort_values('stars', ascending=False)
        for index, row in topic_df.iterrows():
            st.markdown(f'''*Estrellas:* {row['stars']}''')
            st.markdown(f'''*Review:* {row['review']}''')
            st.markdown(f'''---''')

    fig = go.Figure()
    _df = place_text_df[place_text_df['topic'] == -1]
    fig.add_trace(go.Scatter(x=_df['x'], y=_df['y'], hovertext=_df['review'], hoverinfo='text', mode='markers+text', name='Sin tópico', marker=dict(color='#CFD8DC', size=5, opacity=0.5), showlegend=False))
    all_topics = sorted(place_text_df['topic'].unique())
    for topic in all_topics:
        if int(topic) == -1:
            continue
        selection = place_text_df[place_text_df['topic'] == topic]
        label_name = dict_topic_labels[topic]
        fig.add_trace(
            go.Scatter(x=selection['x'], y=selection['y'], hovertext=selection['review'], hoverinfo='text',
                       mode='markers+text', name=label_name, marker=dict(size=5, opacity=0.5)))
    x_range = [place_text_df['x'].min() - abs(place_text_df['x'].min() * 0.15), place_text_df['x'].max() + abs(place_text_df['x'].max() * 0.15)]
    y_range = [place_text_df['y'].min() - abs(place_text_df['y'].min() * 0.15), place_text_df['y'].max() + abs(place_text_df['y'].max() * 0.15)]
    fig.add_shape(type="rect", x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1], line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="rect", x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2, line=dict(color="#CFD8DC", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(x=sum(x_range) / 2, y=y_range[1], text="D2", showarrow=False, xshift=10)
    fig.update_layout(template='simple_white', title={'text': "<b>", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22, color='Black')})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(width=WIDTH * 1.5, height=HEIGHT * 1.5)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    texto_buscar = st.text_input('Ingrese un texto para buscar Tópicos', 'Mejores comidas peruanas')
    if st.button('Buscar'):
        text_emb = st.session_state['sentence_object'].encode(texto_buscar)
        sims = cosine_similarity(text_emb.reshape(1, -1), bertopic_embeddings).flatten()
        sim_ids = np.argsort(sims)[-5:][::-1]
        similarity = [sims[i] for i in sim_ids]
        for enum, index in enumerate(sim_ids):
            similitud = similarity[enum]
            if similitud > 0.5:
                st.markdown(f'''*Tópico:* {dict_topic_labels[index]}''')
                st.markdown(f'''*Similitud:* {similitud}''')

                topic_text_df = text_df[text_df['topic'] == index]
                topic_count_sentiment_df = topic_text_df.groupby('place_id').agg({'id': 'count', 'negative_score': 'mean', 'neutral_score': 'mean', 'positive_score': 'mean'}).reset_index()

                st.markdown(f'''### Cantidad, raking y sentimiento por Place_id ''')
                tab_3_1, tab_3_2 = st.tabs(['Cantidad', 'Ranking'])
                with tab_3_1:
                    topic_count_sentiment_df['id'] = topic_count_sentiment_df['id'] * FACTOR
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=topic_count_sentiment_df['place_id'], y=topic_count_sentiment_df['id'], name='Cantidad de textos', mode='lines+markers'), secondary_y=True)
                    fig.add_trace(go.Bar(x=topic_count_sentiment_df['place_id'], y=topic_count_sentiment_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
                    fig.add_trace(go.Bar(x=topic_count_sentiment_df['place_id'], y=topic_count_sentiment_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
                    fig.add_trace(go.Bar(x=topic_count_sentiment_df['place_id'], y=topic_count_sentiment_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
                    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos por place_id')
                    fig.update_layout(width=WIDTH * 1.5, height=HEIGHT)
                    st.plotly_chart(fig, use_container_width=True)

                with tab_3_2:
                    ranking_date_df = topic_text_df.groupby('place_id')[['stars']].mean().reset_index()
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace( go.Scatter(x=topic_count_sentiment_df['place_id'], y=topic_count_sentiment_df['id'], name='Cantidad de textos', mode='lines+markers'), secondary_y=True)
                    fig.add_trace(go.Bar(x=ranking_date_df['place_id'], y=ranking_date_df['stars'], name='Ranking'), secondary_y=False)
                    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos por tópico')
                    fig.update_layout(width=WIDTH * 1.5, height=HEIGHT)
                    st.plotly_chart(fig, use_container_width=True)

                order_places = topic_count_sentiment_df.sort_values('id', ascending=False)['place_id'].values[:5]
                for place in order_places:
                    st.markdown(f'''#### {place}''')
                    place_df = topic_text_df[topic_text_df['place_id'] == place].head(2)
                    place_df = place_df.sort_values('stars', ascending=False)
                    for index, row in place_df.iterrows():
                        st.markdown(f'''*Estrellas:* {row['stars']}''')
                        st.markdown(f'''*Review:* {row['review']}''')
                        st.markdown(f'''---''')

                st.markdown(f'''---''')


with tab4:
    place_id_1 = st.selectbox('Elige Place_id 1', text_df['place_id'].unique().tolist())
    place_id_2 = st.selectbox('Elige Place_id 2', text_df['place_id'].unique().tolist())
    if st.button('Comparar'):
        if place_id_1 == place_id_2:
            st.markdown(f'''Elige dos place_id diferentes''')
        else:
            place_1_text_df = text_df[text_df['place_id'] == place_id_1]
            place_2_text_df = text_df[text_df['place_id'] == place_id_2]
            st.markdown(f'''### Estadisticas generales ''')
            col_1, col_2 = st.columns(2)
            with col_1:
                st.markdown(f'''#### {place_id_1}''')
                st.markdown(
                    f'''
                    * Número de textos publicados: {place_1_text_df.shape[0] * FACTOR}
                    * Ranking promedio: {place_1_text_df['stars'].mean().round(2)}
                    * Sentimiento negativo: {place_1_text_df['negative_score'].mean().round(2) * 100}%
                    * Sentimiento neutral: {place_1_text_df['neutral_score'].mean().round(2) * 100}%
                    * Sentimiento positivo: {place_1_text_df['positive_score'].mean().round(2) * 100}%
                    ''')
            with col_2:
                st.markdown(f'''#### {place_id_2}''')
                st.markdown(
                    f'''
                    * Número de textos publicados: {place_2_text_df.shape[0] * FACTOR}
                    * Ranking promedio: {place_2_text_df['stars'].mean().round(2)}
                    * Sentimiento negativo: {place_2_text_df['negative_score'].mean().round(2) * 100}%
                    * Sentimiento neutral: {place_2_text_df['neutral_score'].mean().round(2) * 100}%
                    * Sentimiento positivo: {place_2_text_df['positive_score'].mean().round(2) * 100}%
                    ''')

            st.markdown(f'''### Topicos Discriminantes ''')
            places_topicos = list(set(place_1_text_df['topic'].unique().tolist() + place_2_text_df['topic'].unique().tolist()))

            topicos_relevantes_1 = []
            topicos_relevantes_2 = []
            for place_topico in places_topicos:
                if place_topico < 0:
                    continue
                topico_place_1 = sum(place_1_text_df['topic'] == place_topico)
                topico_place_2 = sum(place_2_text_df['topic'] == place_topico)
                chi2, p_val, _, _ = chi2_contingency([[topico_place_1, len(place_1_text_df) - topico_place_1],
                                                      [topico_place_2, len(place_2_text_df) - topico_place_2]])
                if p_val < 0.05:
                    if topico_place_1 > topico_place_2:
                        topicos_relevantes_1.append(place_topico)
                    else:
                        topicos_relevantes_2.append(place_topico)

            if len(topicos_relevantes_1) + len(topicos_relevantes_2) == 0:
                st.markdown(f'''No hay topicos discriminantes''')
            else:
                col_1, col_2 = st.columns(2)
                with col_1:
                    # st.markdown(f'''#### {place_id_1}''')
                    for topico in topicos_relevantes_1:
                        st.markdown(f'''* {dict_topic_labels[topico]}''')
                        place_df = place_1_text_df[place_1_text_df['topic'] == topico].head(2)
                        place_df = place_df.sort_values('stars', ascending=False)
                        for index, row in place_df.iterrows():
                            st.markdown(f'''** Estrellas:* {row['stars']}''')
                            st.markdown(f'''** Review:* {row['review']}''')
                            st.markdown(f'''---''')
                        st.markdown(f'''---''')
                with col_2:
                    # st.markdown(f'''#### {place_id_2}''')
                    for topico in topicos_relevantes_2:
                        st.markdown(f'''* {dict_topic_labels[topico]}''')
                        place_df = place_2_text_df[place_2_text_df['topic'] == topico].head(2)
                        place_df = place_df.sort_values('stars', ascending=False)
                        for index, row in place_df.iterrows():
                            st.markdown(f'''** Estrellas:* {row['stars']}''')
                            st.markdown(f'''** Review:* {row['review']}''')
                            st.markdown(f'''---''')
                        st.markdown(f'''---''')

                places_df = pd.concat([place_1_text_df, place_2_text_df], axis=0).reset_index(drop=True)
                places_df['topic'] = [x if x in list(set(topicos_relevantes_1 + topicos_relevantes_2)) else -1 for x in places_df['topic']]
                fig = go.Figure()
                _df = places_df[places_df['topic'] == -1]
                fig.add_trace(go.Scatter(x=_df['x'], y=_df['y'], hovertext=_df['review'], hoverinfo='text', mode='markers+text', name='Sin tópico', marker=dict(color='#CFD8DC', size=5, opacity=0.5), showlegend=False))
                all_topics = sorted(places_df['topic'].unique())
                for topic in all_topics:
                    if int(topic) == -1:
                        continue
                    selection = places_df[places_df['topic'] == topic]
                    label_name = dict_topic_labels[topic]
                    fig.add_trace(
                        go.Scatter(x=selection['x'], y=selection['y'], hovertext=selection['review'], hoverinfo='text',
                                   mode='markers+text', name=label_name, marker=dict(size=5, opacity=0.5)))
                x_range = [places_df['x'].min() - abs(places_df['x'].min() * 0.15), places_df['x'].max() + abs(places_df['x'].max() * 0.15)]
                y_range = [places_df['y'].min() - abs(places_df['y'].min() * 0.15), places_df['y'].max() + abs(places_df['y'].max() * 0.15)]
                fig.add_shape(type="rect", x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                              line=dict(color="#CFD8DC", width=2))
                fig.add_shape(type="rect", x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                              line=dict(color="#CFD8DC", width=2))
                fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
                fig.add_annotation(x=sum(x_range) / 2, y=y_range[1], text="D2", showarrow=False, xshift=10)
                fig.update_layout(template='simple_white',
                                  title={'text': "<b>", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
                                         'font': dict(size=22, color='Black')})
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)
                fig.update_layout(width=WIDTH * 1.5, height=HEIGHT * 1.5)
                st.plotly_chart(fig, use_container_width=True)

