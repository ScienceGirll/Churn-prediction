import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import plotly.figure_factory as ff
import io













# Ustawienia strony Streamlit
st.set_page_config(layout="wide")

# Wczytanie danych
def load_data():
    df = pd.read_csv(r'C:\Users\weron\OneDrive\Pulpit\wer\dfff.csv')
    return df

# Funkcja do rysowania wykresów słupkowych dla danej zmiennej numerycznej
def draw_bar_charts(selected_col, df, chart_title):
    fig = px.histogram(df, x=selected_col, color_discrete_sequence=['#1f77b4'])
    fig.update_layout(
        xaxis=dict(title='Czas trwania', titlefont=dict(color="#373737")),
        yaxis=dict(title='Liczba klientów', titlefont=dict(color="#373737")),
        plot_bgcolor='#373737',
        paper_bgcolor='#373737',
        font=dict(color='white'),
        width=600,  # Ustawienie szerokości wykresu
        height=500,  # Ustawienie wysokości wykresu
        title=chart_title,  # Dodanie tytułu
        title_font=dict(size=20, color="white"),  # Estetyczny tytuł
        margin=dict(l=20, r=20, t=50, b=20)  # Marginesy
    )
    st.plotly_chart(fig)

# Funkcja do rysowania wykresów kołowych dla danej zmiennej kategorycznej
def draw_pie_charts(selected_col, df, chart_title):
    fig = px.pie(df[selected_col].value_counts(), values=df[selected_col].value_counts(), names=df[selected_col].value_counts().index, 
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        plot_bgcolor='#373737',
        paper_bgcolor='#373737',
        font=dict(color='white'),
        width=600,  # Ustawienie szerokości wykresu
        height=500,  # Ustawienie wysokości wykresu
        title=chart_title,  # Dodanie tytułu
        title_font=dict(size=20, color="white"),  # Estetyczny tytuł
        margin=dict(l=20, r=20, t=50, b=20)  # Marginesy
    )
    st.plotly_chart(fig)

# Funkcja do rysowania separatora
def draw_separator():
    st.write('---')

# Przetwarzanie danych
def preprocess_data(df):
    # Tenure - grupowanie w przedziały co 5 miesięcy
    df['Tenure_Group'] = pd.cut(df['Tenure'], bins=[-1, 5, 10, 15, 20, 25, 30], labels=['0-5', '6-10', '11-15', '16-20', '21-25', '26-30'])

    # WarehouseToHome - grupowanie w przedziały co 5 jednostek
    df['WarehouseToHome_Group'] = pd.cut(df['WarehouseToHome'], bins=[-1, 5, 10, 15, 20, 25, 30, 35, 40], labels=['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40'])

    # CashbackAmount - grupowanie w przedziały co 200
    df['CashbackAmount_Group'] = pd.cut(df['CashbackAmount'], bins=[-1, 200, 400, 600, 800, 1000], labels=['0-200', '201-400', '401-600', '601-800', '801-1000'])

    return df

# Główna część aplikacji Streamlit
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox("Go to", ('Basic info', 'Correlations', 'Prediction'), key="navigation_select")
    st.sidebar.markdown('<style>.sidebar-content { width: 50px }</style>', unsafe_allow_html=True)
    


    if page == 'Basic info':
        st.markdown("""
    <style>
    .fancy-title {
        text-align: center;
        color: yellow;
        font-size: 36px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        position: relative;
    }
    .fancy-title::before {
        content: "";
        position: absolute;
        top: 50%;
        left: -30px;
        width: 0;
        height: 0;
        border-top: 20px solid transparent;
        border-bottom: 20px solid transparent;
        border-right: 30px solid yellow; /* Kolor strzałki */
    }
    </style>
    """, unsafe_allow_html=True)

    # Wyświetl tytuł
        st.markdown("<div class='fancy-title'>Want to see more than just prediction?Distover navigation panel</div>", unsafe_allow_html=True)
        # Wczytaj dane
        df = load_data()
        
        # Usunięcie kolumny CustomerID
        df.drop(columns=['CustomerID'], inplace=True)

        # Przetworzenie danych
        df = preprocess_data(df)

        # Dodatkowy filtr dla danych churn=1, churn=0 lub wszystkich danych
        churn_filter = st.sidebar.radio("choose data for", ('Churn=1', 'Churn=0', 'All'))

        if churn_filter == 'Churn=1':
            df_filtered = df[df['Churn'] == 1]
        elif churn_filter == 'Churn=0':
            df_filtered = df[df['Churn'] == 0]
        else:
            df_filtered = df

        # Zmienna numeryczne i kategoryczne
        numerical_cols = df_filtered.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df_filtered.select_dtypes(exclude=['number']).columns.tolist()




        st.sidebar.title('List of categorical variables')
        selected_categorical_cols = st.sidebar.multiselect('Choose categorical variables', categorical_cols)

        for col in selected_categorical_cols:
            st.title(f'Categorical variable: {col}')
            draw_pie_charts(col, df_filtered, f'Plot - {col}')
            draw_separator()

        st.sidebar.title('List of numerical variables')
        selected_numerical_cols = st.sidebar.multiselect('Choose numerical variables', numerical_cols)

        for col in selected_numerical_cols:
            st.title(f'Numerical variable: {col}')
            draw_bar_charts(col, df_filtered, f'Plot - {col}')

    elif page == 'Correlations':
        st.title('Correlations')
        st.markdown("""
        <style>
        .centered-title {
            text-align: center;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

        # Wczytaj dane
        df = load_data()
        
        # Usunięcie kolumny CustomerID
        df.drop(columns=['CustomerID'], inplace=True)

        # Przetworzenie danych
        df = preprocess_data(df)

        st.subheader('')
        st.subheader('')

        # Dodatkowy filtr dla danych churn=1, churn=0 lub wszystkich danych
        churn_filter = st.radio("Choose data for", ('Churn=1', 'Churn=0', 'Wszystkie'))

        if churn_filter == 'Churn=1':
            df_filtered = df[df['Churn'] == 1]
        elif churn_filter == 'Churn=0':
            df_filtered = df[df['Churn'] == 0]
        else:
            df_filtered = df

        # Layout kolumnowy
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df_filtered, x='Tenure', color_discrete_sequence=['#8da0cb'])
            fig.update_layout(
                xaxis=dict(title='Duration', titlefont=dict(color="white")),
                yaxis=dict(title='Number of clients', titlefont=dict(color="white")),
                plot_bgcolor='#373737',
                paper_bgcolor='#373737',
                font=dict(color='white'),
                width=800,  # Ustawienie szerokości wykresu
                height=500,  # Ustawienie wysokości wykresu
                title='Number of clients in time',  # Dodanie tytułu
                title_font=dict(size=20, color="white"),  # Estetyczny tytuł
                margin=dict(l=20, r=20, t=50, b=20)  # Marginesy
            )
            st.plotly_chart(fig)

            fig = px.bar(df_filtered.groupby('Tenure_Group')['CashbackAmount'].mean().reset_index(), 
                         x='Tenure_Group', y='CashbackAmount', color_discrete_sequence=['#7fc97f'])
            fig.update_layout(
                xaxis=dict(title='Tenure', titlefont=dict(color="white")),
                yaxis=dict(title='Average cashback amount', titlefont=dict(color="white")),
                plot_bgcolor='#373737',
                paper_bgcolor='#373737',
                font=dict(color='white'),
                width=800,  # Ustawienie szerokości wykresu
                height=500,  # Ustawienie wysokości wykresu
                title='Average cashback per Tenure',  # Dodanie tytułu
                title_font=dict(size=20, color="white"),  # Estetyczny tytuł
                margin=dict(l=20, r=20, t=50, b=20)  # Marginesy
            )
            st.plotly_chart(fig)

            fig = px.bar(df_filtered[df_filtered['Complain'] == 1].groupby('CouponUsed').size().reset_index(), 
                         x='CouponUsed', y=0, color_discrete_sequence=['#e78ac3'])
            fig.update_layout(
                xaxis=dict(title='Number of used coupons', titlefont=dict(color="white")),
                yaxis=dict(title='Complain number (Complain=1)', titlefont=dict(color="white")),
                plot_bgcolor='#373737',
                paper_bgcolor='#373737',
                font=dict(color='white'),
                width=800,  # Ustawienie szerokości wykresu
                height=500,  # Ustawienie wysokości wykresu
                title='Complains numbers per used coupons',  # Dodanie tytułu
                title_font=dict(size=20, color="white"),  # Estetyczny tytuł
                margin=dict(l=20, r=20, t=50, b=20)  # Marginesy
            )
            st.plotly_chart(fig)
        
        with col2:
            fig = px.bar(df_filtered.groupby('OrderCount')['CouponUsed'].mean().reset_index(), 
                         x='OrderCount', y='CouponUsed', color_discrete_sequence=['#1f77b4'])
            fig.update_layout(
                xaxis=dict(title='Order count', titlefont=dict(color="white")),
                yaxis=dict(title='Average coupons used per client', titlefont=dict(color="white")),
                plot_bgcolor='#373737',
                paper_bgcolor='#373737',
                font=dict(color='white'),
                width=800,  # Ustawienie szerokości wykresu
                height=500,  # Ustawienie wysokości wykresu
                title='Average coupon use per order count',  # Dodanie tytułu
                title_font=dict(size=20, color="white"),  # Estetyczny tytuł
                margin=dict(l=20, r=20, t=50, b=20)  # Marginesy
            )
            st.plotly_chart(fig)

            fig = px.bar(df_filtered.groupby('PreferredLoginDevice')['HourSpendOnApp'].mean().reset_index(), 
                         x='PreferredLoginDevice', y='HourSpendOnApp', color_discrete_sequence=['#ff7f0e'])
            fig.update_layout(
                xaxis=dict(title='Preferred login device', titlefont=dict(color="white")),
                yaxis=dict(title='Average time spend on app', titlefont=dict(color="white")),
                plot_bgcolor='#373737',
                paper_bgcolor='#373737',
                font=dict(color='white'),
                width=800,  # Ustawienie szerokości wykresu
                height=500,  # Ustawienie wysokości wykresu
                title='Average time on app per preferred device',  # Dodanie tytułu
                title_font=dict(size=20, color="white"),  # Estetyczny tytuł
                margin=dict(l=20, r=20, t=50, b=20)  # Marginesy
            )
            st.plotly_chart(fig)

      
    

    elif page == 'Predication':
        st.title('Features selection and prediction')

# Wczytanie danych
df = load_data()

# Zdefiniuj zmienne numeryczne i kategoryczne
numerical_variables = [
        'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 
        'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress', 
        'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 
        'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
    ]
categorical_variables = {
        'PreferredLoginDevice': ['Mobile Phone', 'Phone', 'Computer'],
        'PreferredPaymentMode': ['Debit Card', 'UPI', 'CC', 'Cash on Delivery', 'E wallet', 'COD', 'Credit Card'],
        'Gender': ['Female', 'Male'],
        'PreferedOrderCat': ['Laptop & Accessory', 'Mobile', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'],
        'MaritalStatus': ['Single', 'Divorced', 'Married']
    }

    # Zbieranie wartości dla zmiennych numerycznych
numerical_values = {}
for variable in numerical_variables:
        value = st.sidebar.text_input(f'Insert value: {variable}')
        if value:
            numerical_values[variable] = float(value)  # Konwertuj na float

    # Zbieranie wartości dla zmiennych kategorycznych
categorical_values = {}
for variable, categories in categorical_variables.items():
        value = st.sidebar.selectbox(f'Insert value: {variable}', options=categories)
        categorical_values[variable] = value

# Utwórz DataFrame z wybranych wartości
data = {**numerical_values, **categorical_values}
df_selected = pd.DataFrame([data])

# Wyświetl DataFrame na stronie
##st.markdown("<h2 style='text-align: center; color: white;'>Choosen values:</h2>", unsafe_allow_html=True)
#st.dataframe(df_selected)
# Formatowanie DataFrame
styled_df_selected = df_selected.style.format({
    # Formatowanie dla kolumn liczbowych: bez miejsc po przecinku
    col: '{:.0f}' if df_selected[col].dtype == 'float64' else '{:}' 
    for col in df_selected.columns
}).set_table_attributes("class='styled-table'").hide_index()

# CSS style
# CSS style
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: white;
        font-size: 36px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        background: linear-gradient(90deg, #5A6F8E, #2C3E50, #BDC3C7);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .container {
        display: flex;
        justify-content: center;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Wyświetl tytuł
st.markdown("<div class='title'>Selected Values</div>", unsafe_allow_html=True)

# Pusty prostokąt pod tytułem
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

st.write(
    f"<div style='display: flex; justify-content: center; flex-direction: column; align-items: center;'>"
    f"{styled_df_selected.render()}</div>", 
    unsafe_allow_html=True
)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
# Przycisk predykcji na głównej stronie
if st.button('Make prediction'):
         st.subheader('')

# Zakładając, że X i y są już zdefiniowane na podstawie wcześniej wczytanych danych df
X = df.drop(columns=['Churn'])
X=X.drop(columns=['CashbackAmountBins'])
X=X.drop(columns=['CustomerID'])
y = df['Churn']



    # Podział danych na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dodanie ostatniego wiersza do ramki danych X
#X_test1 = pd.concat([X_test, df_selected], ignore_index=True)

# Dodanie nowego wiersza do zbioru testowego
#X_test = pd.concat([X_test, df_selected], ignore_index=True)
    # Wybór kolumn numerycznych i kategorycznych
numer = X_train.select_dtypes(include=['int64', 'float64']).columns
categ = X_train.select_dtypes(include=['object']).columns

    # Definicja preprocesora
numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numer),
            ('cat', categorical_transformer, categ)
        ]
    )

    # Definicja modelu XGBoost
    # Definicja rozszerzonej siatki hiperparametrów do przeszukania
param_distributions = {
        'model__n_estimators': [100],
        'model__max_depth': [3],
        'model__learning_rate': [0.01],
        'model__subsample': [0.6],
        'model__colsample_bytree': [0.6],
        'model__gamma': [0]
    }

    # Inicjalizacja modelu XGBoost
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    # Tworzenie pipeline z modelem i SMOTE
pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', xgb)
    ])

    # Inicjalizacja RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, n_iter=20, cv=2, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)

    # Dopasowanie RandomizedSearchCV do danych treningowych
random_search.fit(X_train, y_train)

 # Najlepsze parametry


 # Najlepsze parametry

 # Najlepsze parametry
#st.markdown("<h2 style='text-align: center; color: white;'>Best parameters of model XGBoost:</h2>", unsafe_allow_html=True)

# Przekształcenie słownika najlepszych parametrów na DataFrame
df_best_params = pd.DataFrame({"Parameter": list(random_search.best_params_.keys()), "Value": list(random_search.best_params_.values())})

# Usunięcie indeksów
df_best_params.reset_index(drop=True, inplace=True)

# Wyświetlenie tabeli z najlepszymi parametrami bez kolumny indeksów
# Wyświetlenie tabeli z wybranymi kolumnami
#st.table(df_best_params[["Parametr", "Wartość"]].style.hide_index())
#st.write(df_best_params[["Parameter", "Value"]].to_markdown(index=False), unsafe_allow_html=True)

# Stylizacja DataFrame dla tabeli z najlepszymi parametrami
styled_df_best_params = df_best_params.style.set_table_attributes("class='styled-table'").hide_index()

# Wyświetlenie stylizowanej tabeli z najlepszymi parametrami
#st.write(styled_df_best_params, unsafe_allow_html=True)

    # Najlepszy model
best_model = random_search.best_estimator_

    # Ocena najlepszego modelu na zbiorze testowym
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
#Accuracy

# Pokazanie dokładności w sposób bardziej czytelny z kolorem
if accuracy >= 0.8:
    color = "green"  # Zielony dla wysokiej dokładności
elif accuracy >= 0.6:
    color = "yellow"  # Żółty dla średniej dokładności
else:
    color = "red"  # Czerwony dla niskiej dokładności


# Dodanie stylu CSS do Streamlit
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: white;
        font-size: 36px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        background: linear-gradient(135deg, #485563, #29323c);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1); /* cień */
    }
    .container {
        display: flex;
        justify-content: center;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Wyświetlenie tytułu "Selected Values" z zastosowanym stylem CSS
st.markdown("<div class='title'>Accuracy on test set</div>", unsafe_allow_html=True)
# Pusty prostokąt pod tytułem
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


#st.markdown("<h2 style='text-align: center; color: white;'>Accuracy on test set:</h2>", unsafe_allow_html=True)

# Wyświetlenie dokładności za pomocą st.markdown z odpowiednim kolorem
# Początek markdowa z pierwszego kawałka kodu, który wyświetla dokładność
#st.markdown(
#    f'<div style="text-align: center; font-size:32px;"><span style="color:{color};">{accuracy:.2f}</span></div>',
#    unsafe_allow_html=True
#)


# Styl CSS dla całej strony
st.markdown("""
    <style>
    body {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
        background: linear-gradient(135deg, #f0f0f0, #d0d0d0); /* Gradient białoszary */
    }
    .center-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
    }
    .value-container {
        width: 200px; /* Szerokość kontenera */
        height: 200px; /* Wysokość kontenera */
        background: linear-gradient(to bottom, #ffffff, #f0f0f0); /* Gradient biało-jasnoszary */
        border: 2px solid #ddd; /* Grubość i kolor obramowania */
        border-radius: 10px; /* Zaokrąglenie rogów kontenera */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Cień kontenera dla efektu 3D */
        font-size: 36px; /* Rozmiar czcionki */
        color: #1f78b4; /* Kolor tekstu (można dostosować) */
        font-weight: bold; /* Pogrubienie tekstu */
        display: flex;
        justify-content: center;
        align-items: flex-end;
        padding-bottom: 60px; /* Odstęp na dole */
    }
    </style>
    """, unsafe_allow_html=True)

# Wyświetlenie wartości w stylizowanym kontenerze
st.markdown(
    '<div class="center-container"><div class="value-container">{:.2f}</div></div>'.format(accuracy),
    unsafe_allow_html=True
)

# Tytuł raportu klasyfikacji wyśrodkowany z dodatkowymi marginesami
# Tytuł raportu klasyfikacji wyśrodkowany z dodatkowymi marginesami
# Tytuł raportu klasyfikacji wyśrodkowany z dodatkowymi marginesami
st.markdown("""
    <style>
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            text-align: center;
            color: white;
            margin-top: 50px;
        }
        .styled-table {
            width: 80%;
            border-collapse: collapse;
            border-radius: 5px;
            overflow: hidden;
            margin: auto;
            text-align: center;
        }
        .styled-table th, .styled-table td {
            padding: 12px 15px;
        }
        .styled-table th {
            background-color: #444;
            color: white;
            border-bottom: 2px solid white;
        }
        .styled-table td {
            background-color: #333;
            color: white;
            border-bottom: 1px solid #555;
        }
        .styled-table tr:hover {
            background-color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# Tytuł raportu klasyfikacji wyśrodkowany
#st.markdown("<div class='center'><h2>Classification Report</h2></div>", unsafe_allow_html=True)

# Konwersja raportu klasyfikacji na DataFrame
df_report = pd.DataFrame(report).transpose()

# Wybór tylko wartości klas (0 i 1)
df_report = df_report.loc[['0', '1']]

# Reset indeksu dla lepszej prezentacji
df_report = df_report.reset_index().rename(columns={'index': 'Value'})

# Usuwamy niepotrzebne kolumny
df_report = df_report[['Value', 'precision', 'recall', 'f1-score', 'support']]

# Stylizacja DataFrame
styled_df_report = df_report.style.set_table_attributes("class='styled-table'").hide_index()

# Wyświetlenie stylizowanej tabeli z raportem klasyfikacji
#st.write(styled_df_report, unsafe_allow_html=True)


# Stylizacja DataFrame dla tabeli z najlepszymi parametrami
styled_df_best_params = df_best_params.style.set_table_attributes("class='styled-table'").hide_index()

# Stylizacja DataFrame dla tabeli z raportem klasyfikacji
styled_df_report = df_report.style.set_table_attributes("class='styled-table'").hide_index()



# Najlepsze parametry
#st.markdown("<h2 style='text-align: center; color: white;'>Best parameters of model XGBoost:</h2>", unsafe_allow_html=True)
# Wyświetlenie obu tabel obok siebie w centralnej części strony
# Tytuł nad tabelą z najlepszymi parametrami
# CSS style
# Dodanie stylu CSS do Streamlit
# Dodanie stylu CSS do Streamlit

# Dodanie stylu CSS do Streamlit
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: white;
        font-size: 24px; /* Zmniejszenie rozmiaru czcionki */
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        background: linear-gradient(135deg, #485563, #29323c);
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 0px; /* Zwiększony margines na dole */
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1); /* cień */
    }
    .styled-table {
        border-collapse: collapse;
        width: 100%;
        margin-top: 20px; /* margines między tabelkami */
    }
    </style>
    """, unsafe_allow_html=True)

# Wyświetlenie tytułu "Selected Values"
st.markdown("<div class='title'>Best parameters and classification report</div>", unsafe_allow_html=True)

# Pusty prostokąt pod tytułem
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Stylizacja DataFrame dla tabeli z najlepszymi parametrami
styled_df_best_params = df_best_params.style.set_table_attributes("class='styled-table'").hide_index()

# Stylizacja DataFrame dla tabeli z raportem klasyfikacji
styled_df_report = df_report.style.set_table_attributes("class='styled-table'").hide_index()

# Wyświetlenie obu tabel obok siebie w centralnej części strony, z odpowiednim marginesem pod tytułem
st.write(
    f"<div style='margin-top: 40px; display: flex; justify-content: center; flex-direction: column; align-items: center;'>"
    f"{styled_df_best_params.render()}</div>"
    f"<div style='margin-top: 40px; display: flex; justify-content: center; flex-direction: column; align-items: center;'>"
    f"{styled_df_report.render()}</div>", 
    unsafe_allow_html=True
)

# Pusty prostokąt pod tytułem
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# Pozostała część kodu nie zmienia się

# Pozostała część kodu nie zmienia się


# Define grayscale-blue color palette
colors = [
    [0.0, 'rgb(240, 240, 240)'],     # Very light gray - top left
    [0.33, 'rgb(150, 150, 170)'],    # Light gray - top right
    [0.66, 'rgb(150, 150, 170)'],    # Blue-gray - bottom left
    [1.0, 'rgb(50, 50, 100)']        # Dark blue-gray - bottom right
]

# CSS style
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: white;
        font-size: 36px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        background: linear-gradient(90deg, #5A6F8E, #2C3E50, #BDC3C7);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .container {
        display: flex;
        justify-content: center;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Display title "Confusion Matrix"
st.markdown("<div class='title'>Confusion Matrix</div>", unsafe_allow_html=True)
# Pusty prostokąt pod tytułem
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Create annotated heatmap with custom color palette
fig = ff.create_annotated_heatmap(conf_matrix,
                                 x=['Predicted Negative', 'Predicted Positive'],
                                 y=['Actual Negative', 'Actual Positive'],
                                 colorscale=colors)

# Update layout to increase font sizes and adjust dimensions
fig.update_layout(
    xaxis_title='Predicted label',
    yaxis_title='True label',
    width=1200,     # Set width of the chart
    height=800,     # Set height of the chart
    margin=dict(l=100, r=100, t=100, b=100),  # Adjust margins
    font=dict(
        family="Arial",  # Specify font family
        size=18,         # Increase font size for labels and annotations
        color="black"    # Set font color
    )
)

# Update annotation font size separately (annotations refer to the numbers inside the cells)
for annotation in fig.layout.annotations:
    annotation.font.size = 22  # Adjust font size of numbers inside cells

# Display the plotly chart
st.plotly_chart(fig)
#Best params
# Konwersja słownika best_params_ na obiekt DataFrame
best_params_df = pd.DataFrame.from_dict(random_search.best_params_, orient='index', columns=['Value'])




# Przewidywanie na nowych danych
y_pred_selected = best_model.predict(df_selected)

#st.markdown("<h2 style='text-align: center; color: white;'>Result of our own prediction</h2>", unsafe_allow_html=True)

#if y_pred_selected == 0:
#    st.success("🎉 Good news! This customer is not likely to churn.")
#else:
#    st.error("⚠️ Attention! This customer is likely to churn. Immediate action may be required!")

# Definiowanie kolorów
# Definiowanie kolorów
# Definiowanie kolorów i stylu CSS
background_color = '#F0F0F0'  # Jasnoszary kolor tła
title_color = '#1E88E5'       # Niebieski kolor dla tytułu
success_color = '#4CAF50'     # Zielony kolor dla komunikatu sukcesu
error_color = '#FF5252'       # Czerwony kolor dla komunikatu błędu

# CSS style dla tytułu i komunikatu
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: white;
        font-size: 36px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        background: linear-gradient(90deg, #5A6F8E, #2C3E50, #BDC3C7);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .message-container {
        padding: 20px;
        background-color: transparent;
        border-left: 6px solid;
        border-radius: 5px;
        margin-bottom: 20px;
        animation: fadeIn 1s ease-out;
    }
    .success {
        border-color: """ + success_color + """;
    }
    .error {
        border-color: """ + error_color + """;
    }
    .message {
        font-size: 24px;
        margin-left: 10px;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# Wyświetlenie tytułu w stylu "Confusion Matrix"
st.markdown("<div class='title'>Pediction result</div>", unsafe_allow_html=True)
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
# Symulacja wyniku predykcji (zastąp to swoją logiką rzeczywistą)
y_pred_selected = 1  # Przykład: 0 oznacza brak rezygnacji, 1 oznacza potencjalną rezygnację

# Wyświetlenie komunikatu na podstawie predykcji
if y_pred_selected == 0:
    st.markdown("<div class='message-container success'>"
                "<div class='message'>🎉 Good news! This client probably will not churn.</div>"
                "</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='message-container error'>"
                "<div class='message'>⚠️ Be careful! This client will probably resign!</div>"
                "</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()