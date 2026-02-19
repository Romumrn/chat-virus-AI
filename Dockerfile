# Utilise une image Python légère et officielle
FROM python:3.9-slim

# Définit le répertoire de travail
WORKDIR /app

# Installe les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installe les dépendances Python directement (d'après le README)
RUN pip install --no-cache-dir streamlit pandas numpy requests plotly

# Crée un utilisateur non-root pour des raisons de sécurité
RUN addgroup --system app && adduser --system --ingroup app app

# Copie le reste de l'application
COPY --chown=app:app . .

# Rend le port 8501 accessible (port par défaut de Streamlit)
EXPOSE 8501

# Passe à l'utilisateur non-root
USER app

# Commande pour lancer l'application
CMD ["streamlit", "run", "app.py"]
