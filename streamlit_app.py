import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import arviz as az
import pymc as pm  # en lugar de pymc3


st.title("Análisis Bootstrap y Bayesiano")
st.title("Mi Aplicación de Bootstrap y Análisis Bayesiano")
st.write("Hola, la aplicación se ha cargado correctamente.")


st.write("Sube un dataset con las columnas: **Tanque**, **Atributo** y **Dieta**")
uploaded_file = st.file_uploader("Cargar CSV", type=["csv"])

if uploaded_file is not None:
    # Cargar datos
    df = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.dataframe(df)
    
    # Visualizar la distribución de 'Atributo' por 'Dieta'
    st.subheader("Distribución de Atributo por Dieta")
    fig, ax = plt.subplots()
    sns.boxplot(x="Dieta", y="Atributo", data=df, ax=ax)
    st.pyplot(fig)
    
    # ----------------------------
    # Análisis Bootstrap
    # ----------------------------
    st.subheader("Bootstrap: Distribución de la media de Atributo por Dieta")
    
    n_bootstrap = st.slider("Número de muestras bootstrap", min_value=100, max_value=5000, value=1000, step=100)
    
    # Para cada grupo de 'Dieta', se realizan re-muestreos bootstrap
    bootstrap_results = {}
    for diet in df['Dieta'].unique():
        data_diet = df[df['Dieta'] == diet]['Atributo'].values
        boot_means = [np.mean(np.random.choice(data_diet, size=len(data_diet), replace=True))
                      for _ in range(n_bootstrap)]
        bootstrap_results[diet] = boot_means
        mean_boot = np.mean(boot_means)
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        st.write(f"**Dieta {diet}:** Media bootstrap = {mean_boot:.2f} | Intervalo 95% = [{ci_lower:.2f}, {ci_upper:.2f}]")
        
        fig, ax = plt.subplots()
        sns.histplot(boot_means, kde=True, ax=ax)
        ax.set_title(f"Bootstrap para Dieta: {diet}")
        st.pyplot(fig)
    
    # ----------------------------
    # Análisis Bayesiano
    # ----------------------------
    st.subheader("Análisis Bayesiano")
    st.write("Modelamos *Atributo* como una variable normal para cada grupo de *Dieta*.")
    
    # Convertir 'Dieta' a códigos numéricos para modelar
    df['Dieta_code'] = df['Dieta'].astype('category').cat.codes
    diets = df['Dieta'].astype('category').cat.categories
    st.write("Grupos de Dieta:", diets.tolist())
    
    with pm.Model() as model:
        # Priori para las medias de cada grupo (se asume normal)
        mu = pm.Normal("mu", mu=df['Atributo'].mean(), sigma=10, shape=len(diets))
        # Priori para la desviación estándar (HalfNormal para restringir a valores positivos)
        sigma = pm.HalfNormal("sigma", sigma=10, shape=len(diets))
        
        # Likelihood: cada observación proviene de una normal con la media y sigma del grupo correspondiente
        obs = pm.Normal("obs", mu=mu[df['Dieta_code'].values],
                              sigma=sigma[df['Dieta_code'].values],
                              observed=df['Atributo'].values)
        
        st.write("Ejecutando el muestreo MCMC (esto puede tardar unos segundos)...")
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True, progressbar=True)
    
    st.write("### Resumen del modelo Bayesiano")
    summary = az.summary(trace, var_names=["mu", "sigma"])
    st.write(summary)
    
    st.write("### Distribuciones posteriores de las medias por grupo de Dieta")
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_forest(trace, var_names=["mu"], combined=True, hdi_prob=0.95, ax=ax)
    ax.set_title("Distribución Posterior de las Medias (mu)")
    st.pyplot(fig)
    
    st.write("### Traza del muestreo para un grupo (ejemplo)")
    # Mostrar trazas para el primer grupo
    fig, ax = plt.subplots(figsize=(10, 4))
    az.plot_trace(trace, var_names=["mu"], combined=True, axes=ax)
    st.pyplot(fig)
