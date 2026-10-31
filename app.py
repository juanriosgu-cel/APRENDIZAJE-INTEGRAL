import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve,
    precision_recall_curve, r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import warnings
import hashlib
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Analítica de talento humano", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.25); padding: 14px; border-radius: 12px;}
h1,h2,h3 {letter-spacing: -0.02em;}
.cover-card {background: linear-gradient(120deg, #102A43 0%, #174A6E 58%, #167D8D 100%); color: #FFFFFF; padding: 2rem 2.2rem; border-radius: 18px; margin: .2rem 0 1rem 0; border: 1px solid rgba(255,255,255,.18); box-shadow: 0 8px 24px rgba(15,42,67,.18);}
.cover-kicker {font-size: .78rem; letter-spacing: .12em; font-weight: 700; color: #C9F0F2; margin-bottom: 1rem;}
.cover-card h1 {font-size: clamp(1.8rem, 3vw, 2.65rem); line-height: 1.15; color: white; margin: 0 0 .65rem 0;}
.cover-subtitle {font-size: 1.05rem; color: #EAF7FA; margin: 0;}
.cover-rule {height: 1px; background: rgba(255,255,255,.35); margin: 1.35rem 0;}
.cover-grid {display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem;}
.cover-label {font-size: .72rem; font-weight: 800; letter-spacing: .12em; color: #B9E8EB; margin-bottom: .55rem;}
.cover-names {font-size: .95rem; line-height: 1.9; color: #FFFFFF;}
.cover-teacher {font-size: 1.05rem; font-weight: 650; color: #FFFFFF; padding-top: .25rem;}
@media (max-width: 640px) {.cover-card {padding: 1.35rem;} .cover-grid {grid-template-columns: 1fr; gap: 1rem;}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False, max_entries=1)
def load_data():
    data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    # Normaliza nombres traducidos a los nombres canónicos del dataset IBM.
    # La aplicación trabaja internamente con Attrition; las etiquetas visibles
    # se traducen únicamente en las tablas y visualizaciones.
    reverse_labels = {
        "Edad": "Age", "Abandono": "Attrition",
        "Frecuencia de viajes laborales": "BusinessTravel",
        "Tarifa diaria": "DailyRate", "Departamento": "Department",
        "Distancia desde el hogar": "DistanceFromHome", "Nivel educativo": "Education",
        "Campo de formación": "EducationField", "Conteo de empleados": "EmployeeCount",
        "Número de empleado": "EmployeeNumber",
        "Satisfacción con el entorno laboral": "EnvironmentSatisfaction",
        "Género": "Gender", "Tarifa por hora": "HourlyRate",
        "Involucramiento laboral": "JobInvolvement", "Nivel del cargo": "JobLevel",
        "Cargo": "JobRole", "Satisfacción laboral": "JobSatisfaction",
        "Estado civil": "MaritalStatus", "Ingreso mensual": "MonthlyIncome",
        "Tarifa mensual": "MonthlyRate",
        "Número de empresas en las que ha trabajado": "NumCompaniesWorked",
        "Horas extra": "OverTime",
        "Porcentaje de incremento salarial": "PercentSalaryHike",
        "Calificación de desempeño": "PerformanceRating",
        "Satisfacción con las relaciones laborales": "RelationshipSatisfaction",
        "Horas estándar": "StandardHours",
        "Nivel de opciones sobre acciones": "StockOptionLevel",
        "Experiencia laboral total": "TotalWorkingYears",
        "Capacitaciones realizadas el último año": "TrainingTimesLastYear",
        "Equilibrio entre vida laboral y personal": "WorkLifeBalance",
        "Antigüedad en la empresa": "YearsAtCompany",
        "Antigüedad en el cargo actual": "YearsInCurrentRole",
        "Años desde el último ascenso": "YearsSinceLastPromotion",
        "Antigüedad con el jefe actual": "YearsWithCurrManager",
    }
    data = data.rename(columns={c: reverse_labels[c] for c in data.columns if c in reverse_labels})
    if "Attrition" not in data.columns and "Abandono" in data.columns:
        data = data.rename(columns={"Abandono": "Attrition"})
    if "Attrition" not in data.columns:
        raise ValueError("La base debe contener la variable objetivo 'Attrition' (o su alias 'Abandono').")
    data["Attrition"] = data["Attrition"].replace({
        "Abandonó": "Yes", "No abandonó": "No", "Permaneció": "No",
        "Sí": "Yes"
    })
    return data

def entrenar_random_forest(X_train_data, y_train_data):
    # Parámetros ganadores de GridSearchCV en el notebook ACA1.
    # Se reutiliza el estimador cacheado para evitar entrenamientos duplicados.
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_train_data, y_train_data)
    return modelo

def preparar_datos_clasificacion(dataframe):
    target = "Attrition"
    X = dataframe.drop(
        columns=[target, "EmployeeNumber", "Over18", "StandardHours", "EmployeeCount"],
        errors="ignore"
    )
    y = (dataframe[target] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train_encoded = pd.get_dummies(X_train, drop_first=True, dtype=int)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True, dtype=int).reindex(
        columns=X_train_encoded.columns, fill_value=0
    )
    return X, X_train_encoded, X_test_encoded, y_train, y_test

df = load_data()

# Huella de integridad de la base IBM HR Analytics utilizada en el notebook ACA1.
# Permite detectar si el archivo del despliegue no corresponde al CSV validado.
def huella_base(dataframe):
    payload = pd.util.hash_pandas_object(dataframe, index=True).values.tobytes()
    payload += "|".join(map(str, dataframe.columns)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

HUELLA_BASE_REFERENCIA = "a8b0cee486083ec78743655944f83bde12b0657e61c65c79dc9ded40a654397f"
HUELLA_BASE_ACTUAL = huella_base(df)
BASE_COINCIDE_REFERENCIA = HUELLA_BASE_ACTUAL == HUELLA_BASE_REFERENCIA

# Etiquetas de presentación en español; los nombres internos del dataset permanecen intactos.
EDA_LABELS = {
    "Age":"Edad", "Attrition":"Abandono", "BusinessTravel":"Frecuencia de viajes laborales",
    "DailyRate":"Tarifa diaria", "Department":"Departamento", "DistanceFromHome":"Distancia desde el hogar",
    "Education":"Nivel educativo", "EducationField":"Campo de formación", "EmployeeCount":"Conteo de empleados",
    "EmployeeNumber":"Número de empleado", "EnvironmentSatisfaction":"Satisfacción con el entorno laboral",
    "Gender":"Género", "HourlyRate":"Tarifa por hora", "JobInvolvement":"Involucramiento laboral",
    "JobLevel":"Nivel del cargo", "JobRole":"Cargo", "JobSatisfaction":"Satisfacción laboral",
    "MaritalStatus":"Estado civil", "MonthlyIncome":"Ingreso mensual", "MonthlyRate":"Tarifa mensual",
    "NumCompaniesWorked":"Número de empresas en las que ha trabajado", "OverTime":"Horas extra",
    "PercentSalaryHike":"Porcentaje de incremento salarial", "PerformanceRating":"Calificación de desempeño",
    "RelationshipSatisfaction":"Satisfacción con las relaciones laborales", "StandardHours":"Horas estándar",
    "StockOptionLevel":"Nivel de opciones sobre acciones", "TotalWorkingYears":"Experiencia laboral total",
    "TrainingTimesLastYear":"Capacitaciones realizadas el último año", "WorkLifeBalance":"Equilibrio entre vida laboral y personal",
    "YearsAtCompany":"Antigüedad en la empresa", "YearsInCurrentRole":"Antigüedad en el cargo actual",
    "YearsSinceLastPromotion":"Años desde el último ascenso", "YearsWithCurrManager":"Antigüedad con el jefe actual",
    "Yes":"Sí", "No":"No", "Male":"Masculino", "Female":"Femenino",
    "Research & Development":"Investigación y desarrollo", "Human Resources":"Recursos humanos",
    "Sales":"Ventas", "Life Sciences":"Ciencias de la vida", "Medical":"Ciencias médicas",
    "Marketing":"Mercadeo", "Technical Degree":"Título técnico", "Other":"Otro",
    "Non-Travel":"Sin viajes", "Travel_Rarely":"Viajes ocasionales", "Travel_Frequently":"Viajes frecuentes",
    "Research Scientist":"Científico/a de investigación", "Laboratory Technician":"Técnico/a de laboratorio",
    "Manufacturing Director":"Director/a de manufactura", "Healthcare Representative":"Representante de salud",
    "Sales Executive":"Ejecutivo/a de ventas", "Sales Representative":"Representante de ventas",
    "Human Resources":"Recursos humanos", "Manager":"Gerente", "Director":"Director/a",
    "Single":"Soltero/a", "Married":"Casado/a", "Divorced":"Divorciado/a",
    "Low":"Bajo", "Medium":"Medio", "High":"Alto", "Very High":"Muy alto",
    "No Rating":"Sin calificación", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5"
}

def eda_translate_frame(frame):
    out = frame.copy()
    out.columns = [EDA_LABELS.get(str(c), str(c)) for c in out.columns]
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype) == "category":
            out[col] = out[col].map(lambda v: EDA_LABELS.get(str(v), v) if pd.notna(v) else v)
    return out
st.markdown("""
<div class="cover-card">
  <div class="cover-kicker">APRENDIZAJE AUTOMÁTICO · ESPECIALIZACIÓN EN INTELIGENCIA ARTIFICIAL</div>
  <h1>Analítica exploratoria y modelamiento de talento humano</h1>
  <p class="cover-subtitle">Dataset IBM HR Analytics — Employee Attrition (Abandono)</p>
  <div class="cover-rule"></div>
  <div class="cover-grid">
    <div><div class="cover-label">PRESENTADO POR</div><div class="cover-names">Juan Guillermo Ríos<br>Víctor Rodríguez Polo<br>Michelle Ángel Castillo Mendoza</div></div>
    <div><div class="cover-label">DOCENTE</div><div class="cover-teacher">Hamilton Rivera</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
st.caption("Clasificación del abandono, regresión del ingreso y segmentación de empleados | IBM HR Analytics")
st.info("Aplicación académica para explorar resultados estadísticos. Las asociaciones observadas no demuestran causalidad ni deben utilizarse como única base para tomar decisiones laborales.")

with st.sidebar:
    st.header("Navegación")
    section = st.radio("Seleccione el módulo", [
        "Resumen general", "Análisis exploratorio (EDA)", "Clasificación del abandono", "Regresión del ingreso", "Segmentación de empleados", "Metodología y limitaciones"
    ])
    st.divider()
    st.caption("Base de datos IBM HR Analytics")
    st.write(f"Registros: **{len(df):,}**")
    st.write(f"Variables: **{df.shape[1]}**")
    st.caption(f"Abandono: {(df['Attrition'] == 'Yes').sum():,} | Permanencia: {(df['Attrition'] == 'No').sum():,}")

if section == "Resumen general":
    st.subheader("Resumen de la base de datos")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Empleados", f"{len(df):,}")
    c2.metric("Variables", df.shape[1])
    c3.metric("Abandono", f"{(df['Attrition']=='Yes').mean():.1%}")
    c4.metric("Ingreso mensual promedio", f"${df['MonthlyIncome'].mean():,.0f}")
    st.write("La aplicación integra tres enfoques complementarios sobre la misma base de datos: clasificación para estimar el abandono laboral, regresión para estimar el ingreso mensual y agrupamiento no supervisado para explorar perfiles de empleados.")
    summary_df = df.head(10).copy()
    summary_df.columns = [EDA_LABELS.get(str(c), str(c)) for c in summary_df.columns]
    # Mantener los encabezados del resumen general en español.
    value_translation = {"Sí":"Yes", "No":"No", "Masculino":"Male", "Femenino":"Female",
        "Ventas":"Sales", "Investigación y desarrollo":"Research & Development",
        "Recursos humanos":"Human Resources", "Ciencias de la vida":"Life Sciences",
        "Ciencias médicas":"Medical", "Mercadeo":"Marketing", "Otro":"Other",
        "Viajes ocasionales":"Travel_Rarely", "Viajes frecuentes":"Travel_Frequently", "Sin viajes":"Non-Travel"}
    for col in summary_df.select_dtypes(include=["object"]).columns:
        summary_df[col] = summary_df[col].replace(value_translation)
    st.dataframe(eda_translate_frame(summary_df), use_container_width=True)
    st.subheader("Distribución del abandono")
    counts = df["Attrition"].value_counts().rename_axis("Attrition").reset_index(name="Empleados")
    counts["Attrition"] = counts["Attrition"].replace({"Yes":"Abandonó", "No":"Permaneció"})
    st.plotly_chart(px.bar(counts, x="Attrition", y="Empleados", color="Attrition", title="Distribución del abandono de empleados"), use_container_width=True)
    st.warning("La clase abandono es minoritaria. Por ello, la exactitud por sí sola puede ofrecer una impresión demasiado favorable del desempeño del clasificador.")

elif section == "Clasificación del abandono":
    st.subheader("Clasificación del abandono laboral")
    st.write("Se comparan los cuatro clasificadores con la misma partición estratificada 80/20 y los hiperparámetros seleccionados en el notebook mediante validación cruzada. El entrenamiento es secuencial para controlar el consumo de memoria.")
    X, Xtr_enc, Xte_enc, ytr, yte = preparar_datos_clasificacion(df)

    def entrenar_clasificadores(X_train_data, X_test_data, y_train_data, y_test_data):
        rows, fitted_models = [], {}
        # Hiperparámetros ganadores de GridSearchCV en el notebook ACA1.
        # Se fijan aquí para reproducir sus resultados sin volver a ejecutar
        # una búsqueda exhaustiva en Streamlit Cloud.
        estimators = {
            "Regresión Logística": Pipeline([("escalador", StandardScaler()), ("modelo", LogisticRegression(C=1, penalty="l2", max_iter=2000, random_state=42))]),
            "KNN": Pipeline([("escalador", StandardScaler()), ("modelo", KNeighborsClassifier(n_neighbors=3, metric="manhattan"))]),
            "Árbol de Decisión": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42),
        }
        for name, estimator in estimators.items():
            estimator.fit(X_train_data, y_train_data)
            pred = estimator.predict(X_test_data)
            prob = estimator.predict_proba(X_test_data)[:, 1]
            fitted_models[name] = (estimator, pred, prob)
            rows.append({
                "Modelo": name, "Accuracy": accuracy_score(y_test_data, pred),
                "Precision abandono": precision_score(y_test_data, pred, zero_division=0),
                "Recall abandono": recall_score(y_test_data, pred, zero_division=0),
                "F1 abandono": f1_score(y_test_data, pred, zero_division=0),
                "F1 Macro": f1_score(y_test_data, pred, average="macro", zero_division=0),
                "ROC-AUC": roc_auc_score(y_test_data, prob),
                "PR-AUC": average_precision_score(y_test_data, prob)
            })
        # Random Forest optimizado del notebook, reutilizado desde caché.
        rf_model = entrenar_random_forest(X_train_data, y_train_data)
        rf_pred = rf_model.predict(X_test_data)
        rf_prob = rf_model.predict_proba(X_test_data)[:, 1]
        fitted_models["Random Forest"] = (rf_model, rf_pred, rf_prob)
        rows.append({
            "Modelo": "Random Forest", "Accuracy": accuracy_score(y_test_data, rf_pred),
            "Precision abandono": precision_score(y_test_data, rf_pred, zero_division=0),
            "Recall abandono": recall_score(y_test_data, rf_pred, zero_division=0),
            "F1 abandono": f1_score(y_test_data, rf_pred, zero_division=0),
            "F1 Macro": f1_score(y_test_data, rf_pred, average="macro", zero_division=0),
            "ROC-AUC": roc_auc_score(y_test_data, rf_prob),
            "PR-AUC": average_precision_score(y_test_data, rf_prob)
        })
        return fitted_models, pd.DataFrame(rows).sort_values("F1 abandono", ascending=False)

    fitted, execution_metrics = entrenar_clasificadores(Xtr_enc, Xte_enc, ytr, yte)
    # Métricas y matrices se calculan sobre las predicciones de los modelos
    # configurados con los hiperparámetros del notebook.
    metrics = execution_metrics.copy()
    best = metrics.loc[metrics["Modelo"] == "Random Forest"].iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Modelos comparados", len(metrics))
    c2.metric("Modelo final del estudio", "Random Forest", f"F1 {best['F1 abandono']:.4f}")
    c3.metric("Recall abandono", f"{best['Recall abandono']:.4f}")
    c4.metric("ROC-AUC", f"{best['ROC-AUC']:.4f}")
    st.success(f"**Modelo final del estudio: Random Forest.** En esta ejecución obtuvo F1 de {best['F1 abandono']:.4f} y Recall de {best['Recall abandono']:.4f} para la clase abandono.")
    st.markdown("### Comparación de modelos: configuraciones alineadas con el notebook")
    st.dataframe(metrics.style.format({c:"{:.4f}" for c in metrics.columns if c!="Modelo"}), use_container_width=True, hide_index=True)
    metric_long = metrics.melt(id_vars="Modelo", value_vars=["Accuracy","Precision abandono","Recall abandono","F1 abandono","ROC-AUC","PR-AUC"], var_name="Métrica", value_name="Valor")
    st.plotly_chart(px.bar(metric_long, x="Modelo", y="Valor", color="Métrica", barmode="group", title="Comparación completa de métricas por modelo", text_auto=".3f").update_yaxes(range=[0,1]), use_container_width=True)
    st.markdown("### Análisis detallado del modelo")
    chosen = st.selectbox("Seleccione el modelo que desea analizar", list(fitted.keys()), index=list(fitted.keys()).index(best["Modelo"]), key="class_model_select_v2")
    pipe,pred,prob = fitted[chosen]
    tab1,tab2,tab3,tab4 = st.tabs(["Matriz de confusión", "Curvas ROC y Precision-Recall", "Variables", "PCA"])
    with tab1:
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        st.plotly_chart(px.imshow(cm, text_auto=True, x=["Predijo no abandono","Predijo abandono"], y=["Real no abandono","Real abandono"], color_continuous_scale="Blues", title=f"Matriz de confusión: {chosen}"), use_container_width=True)
        tn,fp,fn,tp = cm.ravel()
        a,b,c,d = st.columns(4)
        a.metric("Verdaderos positivos", int(tp)); b.metric("Falsos negativos", int(fn)); c.metric("Falsos positivos", int(fp)); d.metric("Verdaderos negativos", int(tn))
        st.write(f"**Interpretación:** {tp} abandonos fueron identificados correctamente; {fn} casos reales de abandono no fueron detectados; {fp} empleados fueron clasificados como abandono aunque permanecían. Los falsos negativos son relevantes si se busca identificar tantos abandonos como sea posible.")
        errs=[]
        for nm,(_,pr,_) in fitted.items():
            cmi=confusion_matrix(yte,pr); tni,fpi,fni,tpi=cmi.ravel()
            errs.append({"Modelo":nm,"Verdaderos positivos":tpi,"Falsos negativos":fni,"Falsos positivos":fpi,"Verdaderos negativos":tni})
        errdf=pd.DataFrame(errs)
        st.plotly_chart(px.bar(errdf, x="Modelo", y=["Verdaderos positivos","Falsos negativos","Falsos positivos"], barmode="group", title="Comparación de errores y aciertos entre modelos"), use_container_width=True)
    with tab2:
        figroc=go.Figure()
        figpr=go.Figure()
        for nm,(_,pr,prb) in fitted.items():
            fpr,tpr,_=roc_curve(yte,prb)
            prec,rec,_=precision_recall_curve(yte,prb)
            figroc.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"{nm} (AUC={roc_auc_score(yte,prb):.3f})"))
            figpr.add_trace(go.Scatter(x=rec,y=prec,mode="lines",name=f"{nm} (AP={average_precision_score(yte,prb):.3f})"))
        figroc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Referencia",line=dict(dash="dash")))
        figroc.update_layout(title="Curvas ROC: comparación de todos los modelos",xaxis_title="Tasa de falsos positivos",yaxis_title="Tasa de verdaderos positivos",xaxis=dict(range=[0,1]),yaxis=dict(range=[0,1]))
        figpr.update_layout(title="Curvas Precision-Recall: comparación de todos los modelos",xaxis_title="Recall",yaxis_title="Precision",xaxis=dict(range=[0,1]),yaxis=dict(range=[0,1]))
        st.plotly_chart(figroc,use_container_width=True); st.plotly_chart(figpr,use_container_width=True)
        st.write("**Interpretación:** ROC-AUC resume la capacidad de ordenar casos positivos por encima de negativos. Precision-Recall se enfoca en el desempeño sobre la clase de abandono, minoritaria en esta base.")
    with tab3:
        estimator=pipe.named_steps["modelo"] if hasattr(pipe, "named_steps") else pipe
        names=Xtr_enc.columns.to_numpy()
        if hasattr(estimator,"feature_importances_"):
            imp=pd.DataFrame({"Variable":names,"Importancia":estimator.feature_importances_}).sort_values("Importancia",ascending=False).head(20).sort_values("Importancia")
            st.plotly_chart(px.bar(imp,x="Importancia",y="Variable",orientation="h",title=f"20 variables más importantes: {chosen}"),use_container_width=True)
            st.write("**Interpretación:** estas variables contribuyen más a las decisiones internas del modelo de árbol. La importancia no implica causalidad.")
        elif hasattr(estimator,"coef_"):
            co=np.ravel(estimator.coef_)
            coef=pd.DataFrame({"Variable":names,"Coeficiente":co}).sort_values("Coeficiente",key=abs,ascending=False).head(20).sort_values("Coeficiente")
            st.plotly_chart(px.bar(coef,x="Coeficiente",y="Variable",orientation="h",title=f"Coeficientes principales: {chosen}"),use_container_width=True)
            st.write("**Interpretación:** el signo muestra la dirección de la asociación dentro del modelo y su magnitud depende del preprocesamiento. No representa una relación causal.")
        else:
            st.info("KNN no cuenta con importancia de variables nativa. Para interpretarlo se requiere aplicar una técnica adicional, como permutation importance.")
    with tab4:
        pca=PCA(n_components=2, random_state=42)
        # Aplicar todos los pasos de preprocesamiento del pipeline, sin asumir
        # nombres concretos como "prep" o "escalador".
        if isinstance(pipe, Pipeline) and len(pipe.steps) > 1:
            zmat = pipe[:-1].transform(Xte_enc)
        else:
            zmat = Xte_enc.to_numpy(dtype=float)
        coords=pca.fit_transform(zmat)
        pcadf=pd.DataFrame({"Componente 1":coords[:,0],"Componente 2":coords[:,1],"Clase real":np.where(yte.to_numpy()==1,"Abandonó","Permanece"),"Predicción":np.where(pred==1,"Abandonó","Permanece")})
        st.plotly_chart(px.scatter(pcadf,x="Componente 1",y="Componente 2",color="Clase real",symbol="Predicción",title="PCA de los registros de prueba: clase real y predicción",opacity=.7),use_container_width=True)
        st.caption(f"Los dos componentes explican {pca.explained_variance_ratio_.sum():.1%} de la variabilidad del espacio transformado. Es una proyección para explorar patrones, no una métrica de desempeño.")

    st.markdown("#### Predicción exploratoria de un registro")
    with st.expander("Ingresar valores para estimar probabilidad"):
        vals = {}
        cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
        cols = st.columns(3)
        for i,col in enumerate(X.columns):
            if col in cat:
                opts = sorted(X[col].dropna().astype(str).unique().tolist())
                vals[col] = cols[i%3].selectbox(col, opts, key="cls_"+col)
            else:
                vals[col] = cols[i%3].number_input(col, value=float(X[col].median()), key="cls_"+col, step=1.0)
        if st.button("Estimar probabilidad", key="predict_cls"):
            new_row = pd.DataFrame([vals])
            new_encoded = pd.get_dummies(new_row, drop_first=True, dtype=int).reindex(columns=Xtr_enc.columns, fill_value=0)
            p = pipe.predict_proba(new_encoded)[0,1]
            st.metric("Probabilidad estimada de abandono", f"{p:.1%}")
            st.write("Esta probabilidad es una estimación estadística, no una certeza sobre la conducta futura de una persona.")

elif section == "Regresión del ingreso":
    st.subheader("Regresión del ingreso mensual")
    st.write("Se replica la selección del notebook: ingreso mensual como variable objetivo y edad, antigüedad, abandono, rol y departamento como predictores.")
    data = df[["Attrition","Age","MonthlyIncome","YearsAtCompany","JobRole","Department"]].copy()
    data = data.rename(columns={"Attrition":"Abandono","Age":"Edad","MonthlyIncome":"IngresoMensual","YearsAtCompany":"AñosEnCompañia","JobRole":"RolDeTrabajo","Department":"Departamento"})
    dm = pd.get_dummies(data, columns=["Abandono","RolDeTrabajo","Departamento"], drop_first=True)
    yreg = dm["IngresoMensual"]
    Xreg = dm.drop(columns=["IngresoMensual"])
    Xtr,Xte,ytr,yte = train_test_split(Xreg,yreg,test_size=.2,random_state=42)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    regressors = {"Regresión lineal": (LinearRegression(), Xtr, Xte), "Ridge": (Ridge(alpha=10), Xtr_s, Xte_s), "Lasso": (Lasso(alpha=10,max_iter=10000), Xtr_s, Xte_s)}
    rows=[]; fitted={}; predictions={}
    for name,(model,trainx,testx) in regressors.items():
        model.fit(trainx,ytr); pred=model.predict(testx); fitted[name]=(model,trainx,testx); predictions[name]=pred
        rows.append({"Modelo":name,"R²":r2_score(yte,pred),"RMSE":np.sqrt(mean_squared_error(yte,pred)),"MAE":mean_absolute_error(yte,pred),"MSE":mean_squared_error(yte,pred)})
    met=pd.DataFrame(rows)
    best=met.sort_values(["RMSE","MAE"],ascending=True).iloc[0]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Mejor modelo por RMSE",best["Modelo"])
    c2.metric("R²",f"{best['R²']:.4f}")
    c3.metric("RMSE",f"${best['RMSE']:,.2f}")
    c4.metric("MAE",f"${best['MAE']:,.2f}")
    st.success(f"**Conclusión:** {best['Modelo']} obtiene el menor RMSE en esta ejecución. Su R² es {best['R²']:.3f}, por lo que explica cerca del {best['R²']:.1%} de la variabilidad del ingreso en el conjunto de prueba. La diferencia entre modelos es pequeña; se deben considerar también validación cruzada y residuos.")
    st.markdown("### Comparación de modelos")
    st.dataframe(met.style.format({"R²":"{:.6f}","RMSE":"{:,.2f}","MAE":"{:,.2f}","MSE":"{:,.0f}"}),use_container_width=True,hide_index=True)
    mlong=met.melt(id_vars="Modelo",value_vars=["R²"],var_name="Métrica",value_name="Valor")
    st.plotly_chart(px.bar(mlong,x="Modelo",y="Valor",text_auto=".4f",title="Comparación de R² (mayor es mejor)").update_yaxes(range=[0,1]),use_container_width=True)
    errlong=met.melt(id_vars="Modelo",value_vars=["RMSE","MAE"],var_name="Métrica",value_name="Error")
    st.plotly_chart(px.bar(errlong,x="Modelo",y="Error",color="Métrica",barmode="group",text_auto=".2f",title="Comparación de errores (menor es mejor)"),use_container_width=True)
    st.markdown("### Análisis gráfico por modelo")
    chosen=st.selectbox("Seleccione el modelo para ver sus gráficos e interpretación",["Regresión lineal","Ridge","Lasso"],index=0,key="reg_model_select")
    pred=predictions[chosen]
    real_vs_pred=pd.DataFrame({"Ingreso real":yte.to_numpy(),"Ingreso estimado":pred})
    fig=px.scatter(real_vs_pred,x="Ingreso real",y="Ingreso estimado",opacity=.65,title=f"Ingreso real frente a estimado: {chosen}")
    lo=min(real_vs_pred.min()); hi=max(real_vs_pred.max())
    fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Predicción ideal",line=dict(dash="dash")))
    st.plotly_chart(fig,use_container_width=True)
    resid=yte.to_numpy()-pred
    rdf=pd.DataFrame({"Ingreso estimado":pred,"Residuo":resid})
    st.plotly_chart(px.scatter(rdf,x="Ingreso estimado",y="Residuo",title=f"Diagnóstico de residuos: {chosen}").add_hline(y=0,line_dash="dash"),use_container_width=True)
    st.write(f"**Interpretación de {chosen}:** el MAE es ${met.loc[met['Modelo']==chosen,'MAE'].iloc[0]:,.2f}; en promedio, la diferencia absoluta entre el ingreso real y el estimado tiene ese orden de magnitud. Los residuos deben distribuirse alrededor de cero sin patrones sistemáticos; patrones visibles pueden señalar limitaciones del modelo.")
    model,_,_=fitted[chosen]
    coefs=np.ravel(model.coef_)
    coef=pd.DataFrame({"Variable":Xreg.columns,"Coeficiente":coefs}).sort_values("Coeficiente",key=abs,ascending=False).head(20).sort_values("Coeficiente")
    st.plotly_chart(px.bar(coef,x="Coeficiente",y="Variable",orientation="h",title=f"20 coeficientes de mayor magnitud: {chosen}"),use_container_width=True)
    st.write("**Lectura de coeficientes:** positivos se asocian con un ingreso estimado mayor y negativos con uno menor, manteniendo las demás variables constantes. Para Ridge y Lasso los predictores están estandarizados, por lo que las magnitudes no se interpretan directamente en las unidades originales. No representan causalidad.")

    st.markdown("### Predicción individual del ingreso")
    st.write("Ingrese el perfil de un empleado para estimar su ingreso mensual con el modelo seleccionado arriba.")
    with st.container(border=True):
        pc1, pc2, pc3 = st.columns(3)
        edad_perfil = pc1.number_input("Edad", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()), value=int(df["Age"].median()), step=1, key="reg_pred_age")
        antiguedad_perfil = pc2.number_input("Antigüedad en la empresa (años)", min_value=int(df["YearsAtCompany"].min()), max_value=int(df["YearsAtCompany"].max()), value=int(df["YearsAtCompany"].median()), step=1, key="reg_pred_tenure")
        abandono_perfil = pc3.selectbox("Estado de abandono", sorted(df["Attrition"].dropna().unique().tolist()), key="reg_pred_attrition")
        pc4, pc5 = st.columns(2)
        rol_perfil = pc4.selectbox("Rol de trabajo", sorted(df["JobRole"].dropna().unique().tolist()), key="reg_pred_jobrole")
        depto_perfil = pc5.selectbox("Departamento", sorted(df["Department"].dropna().unique().tolist()), key="reg_pred_department")
        if st.button("Predecir ingreso mensual", key="predict_income"):
            perfil = pd.DataFrame([{
                "Abandono": abandono_perfil,
                "Edad": edad_perfil,
                "AñosEnCompañia": antiguedad_perfil,
                "RolDeTrabajo": rol_perfil,
                "Departamento": depto_perfil,
            }])
            perfil_codificado = pd.get_dummies(perfil, columns=["Abandono", "RolDeTrabajo", "Departamento"], drop_first=True)
            perfil_codificado = perfil_codificado.reindex(columns=Xreg.columns, fill_value=0)
            if chosen == "Regresión lineal":
                prediccion_ingreso = fitted[chosen][0].predict(perfil_codificado)[0]
            else:
                prediccion_ingreso = fitted[chosen][0].predict(scaler.transform(perfil_codificado))[0]
            st.metric("Ingreso mensual estimado", f"${max(0, prediccion_ingreso):,.0f}")
            st.caption(f"Estimación generada con {chosen}. Es una aproximación estadística basada en las variables ingresadas y no garantiza el ingreso real del empleado.")

elif section == "Segmentación de empleados":
    st.subheader("Segmentación y perfiles de empleados")
    st.write("Seleccione las variables y el método para explorar perfiles de empleados. Las variables se estandarizan antes de aplicar los métodos.")

    variables_disponibles = {
        "Edad": "Age",
        "Ingreso mensual": "MonthlyIncome",
        "Experiencia total": "TotalWorkingYears",
        "Antigüedad en la empresa": "YearsAtCompany",
    }
    variables_elegidas = st.multiselect(
        "Variables para segmentar",
        options=list(variables_disponibles.keys()),
        default=list(variables_disponibles.keys()),
        key="seg_variables",
    )
    metodo_explorar = st.selectbox(
        "Método de segmentación",
        ["K-means", "Jerárquico (Ward)", "DBSCAN"],
        index=0,
        key="cluster_method_select",
    )
    if metodo_explorar in ["K-means", "Jerárquico (Ward)"]:
        k_grupos = st.slider("Número de grupos (K)", min_value=2, max_value=10, value=5, step=1, key="seg_k")
    else:
        k_grupos = 5
        col_eps, col_min = st.columns(2)
        eps = col_eps.slider("DBSCAN: distancia (eps)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="seg_eps")
        min_samples = col_min.slider("DBSCAN: mínimo de observaciones", min_value=2, max_value=20, value=5, step=1, key="seg_min_samples")

    if len(variables_elegidas) < 2:
        st.warning("Seleccione al menos dos variables para calcular y comparar la segmentación.")
    else:
        columnas_seg = [variables_disponibles[v] for v in variables_elegidas]
        Xseg = df[columnas_seg].copy()
        Xseg.columns = variables_elegidas
        Z = StandardScaler().fit_transform(Xseg)

        labels_all = {
            "K-means": KMeans(n_clusters=k_grupos, random_state=42, n_init=10).fit_predict(Z),
            "Jerárquico (Ward)": AgglomerativeClustering(n_clusters=k_grupos).fit_predict(Z),
            "DBSCAN": DBSCAN(eps=eps if metodo_explorar == "DBSCAN" else 0.5,
                             min_samples=min_samples if metodo_explorar == "DBSCAN" else 5).fit_predict(Z),
        }
        scores = {}
        for name, lab in labels_all.items():
            # En DBSCAN, Silhouette se calcula sobre los grupos válidos, excluyendo ruido.
            valid = lab != -1 if name == "DBSCAN" else np.ones(len(lab), dtype=bool)
            etiquetas_validas = lab[valid]
            grupos_validos = np.unique(etiquetas_validas)
            if len(grupos_validos) >= 2 and valid.sum() > len(grupos_validos):
                scores[name] = silhouette_score(Z[valid], etiquetas_validas)
            else:
                scores[name] = np.nan

        score_df = pd.DataFrame([
            {
                "Método": nombre,
                "Silhouette": scores[nombre],
                "Grupos": len(set(labels_all[nombre]) - ({-1} if nombre == "DBSCAN" else set())),
                "Ruido": int((labels_all[nombre] == -1).sum()) if nombre == "DBSCAN" else 0,
            }
            for nombre in labels_all
        ])
        valid_scores = score_df.dropna(subset=["Silhouette"])
        best = valid_scores.sort_values("Silhouette", ascending=False).iloc[0] if not valid_scores.empty else None

        c1, c2, c3 = st.columns(3)
        if best is not None:
            c1.metric("Mejor método por Silhouette", best["Método"], f"{best['Silhouette']:.4f}")
            c2.metric("Grupos del mejor método", int(best["Grupos"]))
        else:
            c1.metric("Mejor método", "No calculable")
            c2.metric("Grupos", "—")
        ruido_dbscan = int((labels_all["DBSCAN"] == -1).sum())
        c3.metric("Ruido DBSCAN", f"{ruido_dbscan} ({(labels_all['DBSCAN'] == -1).mean():.2%})")
        if best is not None:
            st.success(
                f"**Mejor método: {best['Método']}.** Obtuvo el mayor coeficiente Silhouette válido "
                f"({best['Silhouette']:.4f}) entre los tres métodos evaluados e identificó "
                f"{int(best['Grupos'])} grupos. La comparación utiliza las variables y los parámetros seleccionados."
            )
        else:
            st.warning("No fue posible calcular un coeficiente Silhouette válido para los métodos evaluados con esta configuración.")

        st.markdown("### Comparación de métodos")
        st.dataframe(score_df.style.format({"Silhouette": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.bar(score_df, x="Método", y="Silhouette", text_auto=".4f", title="Comparación de Silhouette: mayor es mejor"),
            use_container_width=True,
        )
        st.markdown("### Explorar cada método")
        algo = metodo_explorar
        labels = labels_all[algo]
        n_clusters = len(set(labels) - ({-1} if algo == "DBSCAN" else set()))
        noise = int((labels == -1).sum()) if algo == "DBSCAN" else 0
        out = Xseg.copy()
        out["Grupo"] = labels.astype(str)
        profile = out[out["Grupo"] != "-1"].groupby("Grupo")[Xseg.columns].agg(["count", "mean"]).round(2)
        st.markdown(f"#### Perfil de grupos: {algo}")
        st.dataframe(profile, use_container_width=True)
        a, b, c = st.columns(3)
        a.metric("Grupos identificados", n_clusters)
        a_score = scores[algo]
        b.metric("Silhouette", f"{a_score:.4f}" if np.isfinite(a_score) else "No calculable")
        c.metric("Observaciones de ruido", f"{noise:,}" if algo == "DBSCAN" else "No aplica")
        if np.isfinite(a_score):
            st.write("**Interpretación:** " + ("la separación de los grupos es relativamente débil/moderada y existe solapamiento entre algunos perfiles." if a_score < .5 else "los grupos presentan una separación relativamente marcada según Silhouette."))
        if algo == "DBSCAN":
            st.write(f"DBSCAN identificó {n_clusters} grupos y {noise} observaciones como ruido ({noise/len(df):.2%}). El ruido corresponde a registros que no cumplen el criterio de densidad configurado; no significa que sean errores.")
        else:
            st.write("Las etiquetas numéricas de los grupos son arbitrarias. Compare los promedios de las variables seleccionadas para describir los perfiles; no debe inferirse desempeño ni riesgo sin evidencia adicional.")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(Z)
        plot = pd.DataFrame({"Componente 1": coords[:, 0], "Componente 2": coords[:, 1], "Grupo": labels.astype(str)})
        st.plotly_chart(px.scatter(plot, x="Componente 1", y="Componente 2", color="Grupo", title=f"Proyección PCA de empleados: {algo}", opacity=.7), use_container_width=True)
        st.caption(f"Los dos componentes PCA explican conjuntamente {pca.explained_variance_ratio_.sum():.1%} de la varianza estandarizada. La proyección puede ocultar estructura presente en las dimensiones originales.")
        st.markdown("### Comparación visual de los tres métodos")
        fig = go.Figure()
        for method, lab in labels_all.items():
            fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers", name=method, visible=(method == algo), marker=dict(size=6), text=lab.astype(str), hovertemplate="PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>Grupo=%{text}<extra></extra>"))
        fig.update_layout(title="PCA: seleccione la leyenda para mostrar/ocultar métodos", xaxis_title="Componente 1", yaxis_title="Componente 2")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Descargar empleados con grupo asignado", out.to_csv(index=False).encode("utf-8-sig"), f"empleados_segmentados_{algo.replace(' ', '_')}.csv", "text/csv")

elif section == "Análisis exploratorio (EDA)":
    st.subheader("Análisis exploratorio de datos (EDA)")
    st.write("Esta sección consolida los análisis del notebook ACA1_Abandonoaprendizaje.ipynb: estructura y calidad de los datos, distribución del abandono, estadística descriptiva, comparación de variables numéricas y categóricas, correlaciones y detección de valores atípicos. Los resultados son descriptivos y no demuestran causalidad.")

    eda = df.copy()
    eda = eda.drop(columns=["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"], errors="ignore")
    eda_labels = EDA_LABELS
    total = len(eda)
    n_yes = int((eda["Attrition"] == "Yes").sum())
    n_no = int((eda["Attrition"] == "No").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df):,}")
    c2.metric("Variables originales", f"{df.shape[1]}")
    c3.metric("Abandonos", f"{n_yes:,}", f"{n_yes/total:.2%}")
    c4.metric("Permanecen", f"{n_no:,}", f"{n_no/total:.2%}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Calidad de datos", "Distribuciones", "Comparación por abandono", "Relaciones", "Valores atípicos", "Vista de modelos"])
    with tab1:
        st.markdown("### Estructura de la base")
        t1, t2 = st.columns(2)
        t1.write("**Tipos de variables**")
        types = pd.DataFrame({"Variable": df.columns, "Tipo": df.dtypes.astype(str).values, "Valores únicos": [df[c].nunique(dropna=False) for c in df.columns]})
        t1.dataframe(eda_translate_frame(types), use_container_width=True, hide_index=True, height=360)
        t2.write("**Calidad y consistencia**")
        quality = pd.DataFrame({"Valores faltantes": df.isna().sum(), "% faltantes": (df.isna().mean()*100).round(2), "Valores únicos": df.nunique(dropna=False)}).reset_index(names="Variable")
        t2.dataframe(eda_translate_frame(quality), use_container_width=True, hide_index=True, height=360)
        q1,q2,q3 = st.columns(3)
        q1.metric("Celdas faltantes", f"{int(df.isna().sum().sum()):,}")
        q2.metric("Filas duplicadas", f"{int(df.duplicated().sum()):,}")
        q3.metric("Variables constantes", f"{int((df.nunique(dropna=False) <= 1).sum())}")
        checks = {}
        for col in ["YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager", "YearsSinceLastPromotion"]:
            if col in df.columns and "YearsAtCompany" in df.columns:
                checks[f"{col} > YearsAtCompany"] = int((df[col] > df["YearsAtCompany"]).sum()) if col != "YearsAtCompany" else 0
        if checks:
            st.markdown("**Validaciones de consistencia de permanencia**")
            check_df = pd.DataFrame({"Regla revisada": list(checks.keys()), "Registros observados": list(checks.values())})
            check_df["Regla revisada"] = check_df["Regla revisada"].str.replace("YearsAtCompany", EDA_LABELS["YearsAtCompany"]).str.replace("YearsInCurrentRole", EDA_LABELS["YearsInCurrentRole"]).str.replace("YearsWithCurrManager", EDA_LABELS["YearsWithCurrManager"]).str.replace("YearsSinceLastPromotion", EDA_LABELS["YearsSinceLastPromotion"])
            st.dataframe(check_df, use_container_width=True, hide_index=True)
        st.caption("Las variables identificadoras o constantes se excluyen del conjunto analítico cuando no aportan variabilidad; no se eliminan registros automáticamente.")

    with tab2:
        st.markdown("### Distribución del abandono")
        counts = df["Attrition"].value_counts().reindex(["No", "Yes"], fill_value=0)
        dist = pd.DataFrame({"Condición": ["Permanece", "Abandonó"], "Empleados": [int(counts.get("No",0)), int(counts.get("Yes",0))]})
        dist["Porcentaje"] = dist["Empleados"] / max(len(df),1) * 100
        c1,c2 = st.columns([1,1])
        c1.dataframe(dist.style.format({"Porcentaje":"{:.2f}%"}), use_container_width=True, hide_index=True)
        c2.plotly_chart(px.bar(dist, x="Condición", y="Empleados", color="Condición", text="Porcentaje", color_discrete_map={"Permanece":"#2583C5","Abandonó":"#E76F51"}, title="Distribución del abandono", labels={"Porcentaje":"Porcentaje (%)"}), use_container_width=True)
        st.markdown("### Estadística descriptiva")
        num_cols = [c for c in ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears", "DistanceFromHome", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"] if c in df.columns]
        desc = eda[num_cols].describe().T.reset_index(names="Variable")
        st.dataframe(eda_translate_frame(desc), use_container_width=True)
        st.markdown("### Distribuciones numéricas")
        selected_num = st.selectbox("Seleccione una variable numérica", num_cols, format_func=lambda v: EDA_LABELS.get(v, v), key="eda_num_dist")
        eda_hist = df[[selected_num, "Attrition"]].copy()
        eda_hist["Estado de abandono"] = eda_hist["Attrition"].replace({"Yes": "Abandonó", "No": "Permaneció"})
        st.plotly_chart(
            px.histogram(
                eda_hist, x=selected_num, color="Estado de abandono",
                marginal="box", barmode="overlay", opacity=.65,
                title=f"Distribución de {EDA_LABELS.get(selected_num, selected_num)} según abandono",
                labels={**EDA_LABELS, "Estado de abandono": "Estado de abandono"}
            ),
            use_container_width=True
        )

    with tab3:
        st.markdown("### Variables numéricas según condición de abandono")
        num_compare = [c for c in ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears", "DistanceFromHome", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"] if c in df.columns]
        selected_compare = st.multiselect("Variables numéricas para comparar", num_compare, default=[c for c in ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears"] if c in num_compare], format_func=lambda v: EDA_LABELS.get(v, v), key="eda_num_compare")
        if selected_compare:
            medians = df.groupby("Attrition")[selected_compare].median().T.reset_index(names="Variable")
            means = df.groupby("Attrition")[selected_compare].mean().T.reset_index(names="Variable")
            st.markdown("**Medianas por condición**")
            st.dataframe(eda_translate_frame(medians), use_container_width=True, hide_index=True)
            st.markdown("**Medias por condición**")
            st.dataframe(eda_translate_frame(means.round(2)), use_container_width=True, hide_index=True)
            melted = df.melt(id_vars="Attrition", value_vars=selected_compare, var_name="Variable", value_name="Valor")
            st.plotly_chart(px.box(melted, x="Variable", y="Valor", color="Attrition", facet_col="Variable", facet_col_wrap=2, title="Distribuciones numéricas según abandono", points=False, labels=EDA_LABELS), use_container_width=True)
        st.markdown("### Tasa de abandono por categoría")
        cat_candidates = [c for c in ["OverTime", "BusinessTravel", "JobSatisfaction", "EnvironmentSatisfaction", "JobInvolvement", "WorkLifeBalance", "JobLevel", "Department", "JobRole", "MaritalStatus", "StockOptionLevel"] if c in df.columns]
        cat_var = st.selectbox("Seleccione una variable categórica", cat_candidates, index=cat_candidates.index("OverTime") if "OverTime" in cat_candidates else 0, format_func=lambda v: EDA_LABELS.get(v, v), key="eda_cat_rate")
        rate = (df.assign(_abandono=(df["Attrition"]=="Yes").astype(int)).groupby(cat_var, dropna=False)['_abandono'].agg(Empleados="size", Abandonos="sum", Tasa_abandono="mean").reset_index())
        rate["Tasa_abandono_%"] = rate["Tasa_abandono"]*100
        rate = rate.sort_values("Tasa_abandono_%", ascending=False)
        rate_display = eda_translate_frame(rate.drop(columns="Tasa_abandono"))
        st.dataframe(rate_display.style.format({"Tasa_abandono_%":"{:.2f}%"}), use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(rate, x=cat_var, y="Tasa_abandono_%", text="Tasa_abandono_%", title=f"Tasa de abandono según {EDA_LABELS.get(cat_var, cat_var)}", labels={**EDA_LABELS, "Tasa_abandono_%":"Tasa de abandono (%)"}), use_container_width=True)

    with tab4:
        st.markdown("### Abandono según rol y departamento")
        org_var = st.selectbox("Variable organizacional", [c for c in ["Department", "JobRole"] if c in df.columns], format_func=lambda v: EDA_LABELS.get(v, v), key="eda_org_var")
        org = (df.assign(_abandono=(df["Attrition"]=="Yes").astype(int)).groupby(org_var)['_abandono'].agg(Empleados="size", Abandonos="sum", Tasa="mean").reset_index())
        org["Tasa de abandono (%)"] = org["Tasa"]*100
        org = org.sort_values("Tasa de abandono (%)", ascending=False)
        org_display = eda_translate_frame(org.drop(columns="Tasa"))
        st.dataframe(org_display.style.format({"Tasa de abandono (%)":"{:.2f}%"}), use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(org, x=org_var, y="Tasa de abandono (%)", title=f"Tasa de abandono por {EDA_LABELS.get(org_var, org_var)}", text="Tasa de abandono (%)", labels=EDA_LABELS), use_container_width=True)
        st.markdown("### Correlación entre variables numéricas")
        corr_vars = [c for c in ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears", "DistanceFromHome", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"] if c in df.columns]
        corr = df[corr_vars].corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Matriz de correlación de variables numéricas", labels=EDA_LABELS), use_container_width=True)
        st.caption("La correlación describe asociación lineal entre variables numéricas; no demuestra causalidad.")
        st.markdown("### Relación conjunta de variables")
        pair_vars = [c for c in ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears"] if c in df.columns]
        pair_x = st.selectbox("Variable del eje X", pair_vars, index=0, key="eda_pair_x")
        pair_y = st.selectbox("Variable del eje Y", [c for c in pair_vars if c != pair_x], index=0, key="eda_pair_y")
        st.plotly_chart(px.scatter(df, x=pair_x, y=pair_y, color="Attrition", opacity=.65, title=f"{EDA_LABELS.get(pair_y,pair_y)} frente a {EDA_LABELS.get(pair_x,pair_x)}, según abandono", labels=EDA_LABELS), use_container_width=True)
        st.markdown("### Matriz de dispersión de variables principales")
        st.caption("Permite explorar simultáneamente las relaciones entre edad, ingreso mensual, antigüedad y experiencia, diferenciando la condición de abandono.")
        st.plotly_chart(
            px.scatter_matrix(
                df,
                dimensions=pair_vars,
                color="Attrition",
                opacity=0.55,
                title="Matriz de dispersión: variables principales y abandono",
                labels=EDA_LABELS
            ),
            use_container_width=True
        )

    with tab5:
        st.markdown("### Análisis multivariado de valores atípicos")
        st.caption("Los puntos resaltados se identifican con el criterio IQR aplicado a cada variable numérica. Un valor atípico no es necesariamente un error y no debe eliminarse sin revisión contextual.")
        out_vars = [c for c in ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears", "DistanceFromHome", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"] if c in df.columns]
        out_rows=[]
        outlier_flags = pd.Series(False, index=df.index)
        for col in out_vars:
            q1=df[col].quantile(.25); q3=df[col].quantile(.75); iqr=q3-q1
            lo=q1-1.5*iqr; hi=q3+1.5*iqr
            flags=(df[col]<lo)|(df[col]>hi)
            outlier_flags |= flags
            out_rows.append({"Variable":col,"Q1":q1,"Q3":q3,"IQR":iqr,"Límite inferior":lo,"Límite superior":hi,"Valores atípicos":int(flags.sum())})
        out_df=pd.DataFrame(out_rows)
        k1,k2,k3=st.columns(3)
        k1.metric("Variables evaluadas",len(out_vars))
        k2.metric("Registros con algún valor atípico",f"{int(outlier_flags.sum()):,}")
        k3.metric("Criterio aplicado","IQR · 1,5 × rango")
        out_df_display = eda_translate_frame(out_df)
        st.dataframe(out_df_display.style.format({c:"{:,.2f}" for c in ["Q1","Q3","IQR","Límite inferior","Límite superior"]}), use_container_width=True, hide_index=True)
        plot_df=df.copy()
        plot_df["Clasificación IQR"] = np.where(outlier_flags, "Posible valor atípico", "Dentro del rango IQR")
        plot_specs=[("Age","MonthlyIncome","Edad e ingreso mensual"),("MonthlyIncome","YearsAtCompany","Ingreso y antigüedad"),("Age","YearsAtCompany","Edad y antigüedad")]
        plot_specs=[spec for spec in plot_specs if spec[0] in df.columns and spec[1] in df.columns]
        cols=st.columns(len(plot_specs))
        for colbox,(xvar,yvar,title) in zip(cols,plot_specs):
            fig=px.scatter(plot_df,x=xvar,y=yvar,color="Clasificación IQR",color_discrete_map={"Dentro del rango IQR":"#2583C5","Posible valor atípico":"#E76F51"},opacity=.72,title=title,hover_data=["Attrition"])
            fig.update_layout(template="plotly_white",title_x=.03,legend_title_text="Tipo de registro",font=dict(size=12),margin=dict(l=12,r=12,t=55,b=12),height=390)
            colbox.plotly_chart(fig,use_container_width=True)
        out_var=st.selectbox("Explorar una variable en detalle", out_vars, format_func=lambda v: EDA_LABELS.get(v, v), key="eda_outlier_var")
        boxfig=px.box(df,y=out_var,points="outliers",color_discrete_sequence=["#2583C5"],title=f"Distribución y valores extremos: {EDA_LABELS.get(out_var,out_var)}", labels=EDA_LABELS)
        boxfig.update_layout(template="plotly_white",title_x=.03,margin=dict(l=12,r=12,t=55,b=12),height=420)
        st.plotly_chart(boxfig,use_container_width=True)
        st.markdown("#### Criterios de lectura")
        st.markdown("- **Azul:** registros que no presentan valores fuera de los límites IQR en las variables evaluadas.\n- **Coral:** registros con al menos un valor fuera de esos límites.\n- **Importante:** la marca es una alerta exploratoria, no una conclusión de error o anomalía laboral.")

    with tab6:
        st.markdown("### Resultados visuales de los modelos")
        st.caption("Estas vistas complementan los módulos de modelamiento. Las métricas se calculan en una partición reproducible; la segmentación es exploratoria y no implica perfiles de desempeño laboral.")

        # Clasificación: usar exactamente la misma partición, codificación y
        # configuraciones que el módulo Clasificación del abandono.
        st.markdown("#### 1. Clasificación del abandono")
        Xc, Xtrc_enc, Xtec_enc, ytrc, ytec = preparar_datos_clasificacion(df)
        cls_models = {
            "Regresión Logística": Pipeline([("escalador", StandardScaler()), ("modelo", LogisticRegression(C=1, penalty="l2", max_iter=2000, random_state=42))]),
            "KNN": Pipeline([("escalador", StandardScaler()), ("modelo", KNeighborsClassifier(n_neighbors=3, metric="manhattan"))]),
            "Árbol de Decisión": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42),
        }
        cls_rows=[]; cls_fitted={}
        for nm, estimator in cls_models.items():
            estimator.fit(Xtrc_enc, ytrc)
            pred_cls=estimator.predict(Xtec_enc)
            prob_cls=estimator.predict_proba(Xtec_enc)[:,1]
            cls_fitted[nm]=(pred_cls,prob_cls)
            cls_rows.append({"Modelo":nm,"Precision abandono":precision_score(ytec,pred_cls,zero_division=0),"Recall abandono":recall_score(ytec,pred_cls,zero_division=0),"F1 abandono":f1_score(ytec,pred_cls,zero_division=0),"ROC-AUC":roc_auc_score(ytec,prob_cls)})
        # Reutiliza el mismo Random Forest optimizado y cacheado del módulo principal.
        rf_eda = entrenar_random_forest(Xtrc_enc, ytrc)
        pred_rf_eda = rf_eda.predict(Xtec_enc)
        prob_rf_eda = rf_eda.predict_proba(Xtec_enc)[:, 1]
        cls_fitted["Random Forest"] = (pred_rf_eda, prob_rf_eda)
        cls_rows.append({"Modelo":"Random Forest",
            "Precision abandono":precision_score(ytec,pred_rf_eda,zero_division=0),
            "Recall abandono":recall_score(ytec,pred_rf_eda,zero_division=0),
            "F1 abandono":f1_score(ytec,pred_rf_eda,zero_division=0),
            "ROC-AUC":roc_auc_score(ytec,prob_rf_eda)})
        cls_metrics=pd.DataFrame(cls_rows).sort_values("F1 abandono",ascending=False)
        cc1,cc2,cc3=st.columns(3)
        cc1.metric("Modelos comparados",len(cls_metrics))
        cc2.metric("Mayor F1 de abandono",f"{cls_metrics.iloc[0]['F1 abandono']:.3f}",cls_metrics.iloc[0]["Modelo"])
        cc3.metric("Mayor Recall",f"{cls_metrics['Recall abandono'].max():.3f}")
        cls_long=cls_metrics.melt(id_vars="Modelo",value_vars=["Precision abandono","Recall abandono","F1 abandono","ROC-AUC"],var_name="Métrica",value_name="Valor")
        fig_cls=px.bar(cls_long,x="Modelo",y="Valor",color="Métrica",barmode="group",text_auto=".2f",title="Comparación de clasificadores en datos de prueba",color_discrete_sequence=["#176B87","#64CCC5","#E76F51","#6C63A8"])
        fig_cls.update_layout(template="plotly_white",title_x=.02,yaxis=dict(range=[0,1]),legend_title_text="Métrica",margin=dict(l=10,r=10,t=65,b=10),height=440)
        st.plotly_chart(fig_cls,use_container_width=True)
        cls_choice=st.selectbox("Modelo para ver matriz de confusión",list(cls_fitted.keys()),index=list(cls_fitted.keys()).index("Random Forest"),key="eda_cls_matrix_choice")
        cm=confusion_matrix(ytec,cls_fitted[cls_choice][0],labels=[0,1])
        fig_cm=px.imshow(cm,text_auto=True,x=["Predice permanece","Predice abandono"],y=["Real permanece","Real abandono"],color_continuous_scale="Blues",title=f"Matriz de confusión — {cls_choice}")
        fig_cm.update_layout(template="plotly_white",title_x=.02,height=390)
        st.plotly_chart(fig_cm,use_container_width=True)
        st.caption("La matriz utiliza la misma partición estratificada, codificación y configuración de modelos que el módulo de clasificación.")

        # Regresión: predicciones reales de tres modelos en la misma partición.
        st.markdown("#### 2. Regresión del ingreso mensual")
        reg_data=df[["Attrition","Age","MonthlyIncome","YearsAtCompany","JobRole","Department"]].copy()
        reg_data=reg_data.rename(columns={"Attrition":"Abandono","Age":"Edad","MonthlyIncome":"IngresoMensual","YearsAtCompany":"AñosEnCompañia","JobRole":"RolDeTrabajo","Department":"Departamento"})
        reg_dm=pd.get_dummies(reg_data,columns=["Abandono","RolDeTrabajo","Departamento"],drop_first=True)
        yreg_eda=reg_dm["IngresoMensual"]
        Xreg_eda=reg_dm.drop(columns=["IngresoMensual"])
        Xtrr,Xter,ytrr,yter=train_test_split(Xreg_eda,yreg_eda,test_size=.2,random_state=42)
        scaler_eda=StandardScaler(); Xtrrs=scaler_eda.fit_transform(Xtrr); Xters=scaler_eda.transform(Xter)
        reg_defs={"Regresión lineal":(LinearRegression(),Xtrr,Xter),"Ridge":(Ridge(alpha=10),Xtrrs,Xters),"Lasso":(Lasso(alpha=10,max_iter=10000),Xtrrs,Xters)}
        reg_rows=[]; reg_preds={}
        for nm,(est,xtrain,xtest) in reg_defs.items():
            est.fit(xtrain,ytrr); pr=est.predict(xtest); reg_preds[nm]=pr
            reg_rows.append({"Modelo":nm,"R²":r2_score(yter,pr),"RMSE":np.sqrt(mean_squared_error(yter,pr)),"MAE":mean_absolute_error(yter,pr)})
        reg_metrics=pd.DataFrame(reg_rows).sort_values("RMSE")
        rm1,rm2,rm3=st.columns(3)
        rm1.metric("Mejor modelo por RMSE",reg_metrics.iloc[0]["Modelo"])
        rm2.metric("Menor RMSE",f"${reg_metrics.iloc[0]['RMSE']:,.0f}")
        rm3.metric("MAE del mejor",f"${reg_metrics.iloc[0]['MAE']:,.0f}")
        fig_regmetrics=px.bar(reg_metrics,x="Modelo",y="RMSE",color="Modelo",text_auto=".0f",title="Error de predicción del ingreso mensual (menor es mejor)",color_discrete_sequence=["#176B87","#64CCC5","#E76F51"])
        fig_regmetrics.update_layout(template="plotly_white",title_x=.02,showlegend=False,height=370)
        st.plotly_chart(fig_regmetrics,use_container_width=True)
        reg_choice=st.selectbox("Modelo para comparar ingreso real y estimado",list(reg_preds.keys()),key="eda_reg_choice")
        reg_compare=pd.DataFrame({"Ingreso real":yter.to_numpy(),"Ingreso estimado":reg_preds[reg_choice]})
        fig_actual=px.scatter(reg_compare,x="Ingreso real",y="Ingreso estimado",opacity=.62,title=f"Ingreso real frente a estimado — {reg_choice}",color_discrete_sequence=["#176B87"])
        lo=float(min(reg_compare.min())); hi=float(max(reg_compare.max()))
        fig_actual.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Estimación ideal",line=dict(color="#E76F51",dash="dash",width=2)))
        fig_actual.update_layout(template="plotly_white",title_x=.02,height=430)
        st.plotly_chart(fig_actual,use_container_width=True)

        # Segmentación: grupos K-means y perfiles descriptivos.
        st.markdown("#### 3. Segmentación exploratoria de empleados")
        seg_vars=["Age","MonthlyIncome","YearsAtCompany","TotalWorkingYears"]
        seg_z=StandardScaler().fit_transform(df[seg_vars])
        seg_kmeans=KMeans(n_clusters=5,random_state=42,n_init=10)
        seg_lab=seg_kmeans.fit_predict(seg_z)
        seg_pca=PCA(n_components=2,random_state=42)
        seg_coords=seg_pca.fit_transform(seg_z)
        seg_df=pd.DataFrame({"Componente 1":seg_coords[:,0],"Componente 2":seg_coords[:,1],"Grupo":seg_lab.astype(str),"Abandono":df["Attrition"].values})
        fig_seg=px.scatter(seg_df,x="Componente 1",y="Componente 2",color="Grupo",symbol="Abandono",opacity=.72,title="Cinco grupos K-means proyectados mediante PCA",hover_data=["Abandono"],color_discrete_sequence=["#176B87","#64CCC5","#E76F51","#6C63A8","#E9C46A"])
        fig_seg.update_layout(template="plotly_white",title_x=.02,height=480,legend_title_text="Grupo")
        st.plotly_chart(fig_seg,use_container_width=True)
        seg_with_labels=df[seg_vars].copy()
        seg_with_labels["Grupo"]=seg_lab
        seg_profile=seg_with_labels.groupby("Grupo").agg({"Age":"mean","MonthlyIncome":"mean","YearsAtCompany":"mean","TotalWorkingYears":"mean"}).rename(columns={"Age":"Edad media","MonthlyIncome":"Ingreso medio","YearsAtCompany":"Antigüedad media","TotalWorkingYears":"Experiencia media"}).round(1).reset_index()
        seg_counts=seg_with_labels.groupby("Grupo").size().rename("Empleados").reset_index()
        seg_profile=seg_profile.merge(seg_counts,on="Grupo",how="left")
        st.dataframe(seg_profile.style.format({"Ingreso medio":"${:,.0f}"}),use_container_width=True,hide_index=True)
        st.caption(f"La proyección PCA representa {seg_pca.explained_variance_ratio_.sum():.1%} de la varianza estandarizada en dos dimensiones. Los grupos son exploratorios; no constituyen una evaluación de los empleados.")

    st.markdown("### Principales hallazgos de la EDA")
    med = df.groupby("Attrition")[[c for c in ["YearsAtCompany", "MonthlyIncome", "Age"] if c in df.columns]].median()
    overtime_rates = (df.assign(_abandono=(df["Attrition"]=="Yes").astype(int)).groupby("OverTime")['_abandono'].mean()*100) if "OverTime" in df.columns else pd.Series(dtype=float)
    c_yes = float((df["Attrition"]=="Yes").mean()*100)
    st.write(f"- El abandono representa **{c_yes:.2f}%** de los registros ({n_yes:,} empleados).")
    if "YearsAtCompany" in med.columns and "Yes" in med.index and "No" in med.index:
        st.write(f"- La mediana de antigüedad es de **{med.loc['Yes','YearsAtCompany']:.0f} años** entre quienes abandonaron y **{med.loc['No','YearsAtCompany']:.0f} años** entre quienes permanecieron.")
    if "MonthlyIncome" in med.columns and "Yes" in med.index and "No" in med.index:
        st.write(f"- La mediana del ingreso mensual es **${med.loc['Yes','MonthlyIncome']:,.0f}** en el grupo que abandonó y **${med.loc['No','MonthlyIncome']:,.0f}** en el grupo que permaneció.")
    if len(overtime_rates) and "Yes" in overtime_rates.index and "No" in overtime_rates.index:
        st.write(f"- La tasa de abandono es **{overtime_rates['Yes']:.1f}%** entre quienes realizan horas extra y **{overtime_rates['No']:.1f}%** entre quienes no las realizan.")
    if "JobRole" in df.columns:
        role_rates=(df.assign(_abandono=(df["Attrition"]=="Yes").astype(int)).groupby("JobRole")['_abandono'].mean()*100).sort_values(ascending=False)
        if not role_rates.empty:
            st.write(f"- El rol con mayor tasa observada de abandono es **{role_rates.index[0]}** ({role_rates.iloc[0]:.1f}%).")
    st.info("Estos hallazgos son descriptivos. Las diferencias observadas no prueban que una variable cause el abandono y deben evaluarse con análisis estadísticos y modelos posteriores.")

else:
    st.subheader("Metodología y limitaciones")
    etapas = [
        ("1. Datos", "Se utiliza la base IBM HR Analytics, con 1.470 registros y variables demográficas, laborales y salariales."),
        ("2. Preprocesamiento", "Se revisan tipos de variables y se transforman las variables categóricas mediante codificación. En clasificación se aplica codificación dummy alineada entre entrenamiento y prueba; en regresión se emplea la codificación dummy definida en el notebook; en segmentación se estandarizan las cuatro variables numéricas seleccionadas."),
        ("3. Modelamiento", "Clasificación: Regresión Logística, KNN, Árbol de Decisión y Random Forest. Regresión: Lineal, Ridge y Lasso. Segmentación: K-means, jerárquico y DBSCAN."),
        ("4. Optimización y selección", "Los modelos se comparan con métricas apropiadas a cada tarea. Para regresión se consideran R², MAE y RMSE; para segmentación se compara Silhouette; para clasificación se revisan Recall, F1, ROC-AUC y PR-AUC, además de la matriz de confusión."),
        ("5. Evaluación e interpretación", "Cada módulo presenta tablas, visualizaciones y una interpretación automática que identifica el mejor resultado según el criterio definido. La selección se refiere a los modelos y parámetros ejecutados en la aplicación."),
        ("6. Limitaciones", "Los resultados dependen de las variables, particiones y parámetros empleados. Las asociaciones no demuestran causalidad, las etiquetas de grupo son arbitrarias y las predicciones no deben utilizarse como único criterio para decisiones laborales.")
    ]
    for titulo, descripcion in etapas:
        with st.container(border=True):
            st.subheader(titulo)
            st.write(descripcion)
    st.info("Referencia a los resultados de los notebooks: Ridge obtuvo el menor RMSE en la evaluación de regresión registrada (1.943,80 aproximadamente); K-means obtuvo Silhouette de 0,3363, superior al jerárquico (0,3079) y DBSCAN (0,3007). En clasificación, la aplicación recalcula métricas en su partición de prueba y pueden diferir de las cifras del notebook original.")
    st.write("Base usada:", f"{len(df):,} filas y {df.shape[1]} columnas.")
    st.write("Valores faltantes totales:", int(df.isna().sum().sum()))
    st.write("Registros duplicados:", int(df.duplicated().sum()))
