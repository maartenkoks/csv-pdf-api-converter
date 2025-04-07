# CSV/PDF to API Converter & Hoster

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Flask Version](https://img.shields.io/badge/flask-2.3.2-teal.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Beschrijving

Deze Flask-applicatie stelt gebruikers in staat om:

1.  Een **CSV-bestand** te uploaden, te laten analyseren (kolomtypen, voorbeelden), kolom beschrijvingen te laten genereren via OpenAI (GPT-4o-mini), en vervolgens een doorzoekbare **REST API** te genereren voor de data in het CSV-bestand. De API ondersteunt filtering op geselecteerde kolommen en een optionele multi-kolom keyword search.
2.  Een **PDF-bestand** te uploaden en deze te hosten zodat deze publiek toegankelijk is via een directe link.

De applicatie kan lokaal worden gedraaid en maakt gebruik van Ngrok (tijdens lokale ontwikkeling) om de API en gehoste PDF's tijdelijk publiek toegankelijk te maken, wat nuttig is voor integratie met externe diensten zoals HALO. Voor permanent gebruik is deployment naar een hosting platform nodig.

## Features

*   **CSV Upload & Analyse:** Robuuste verwerking van diverse CSV-bestanden (detecteert encoding en delimiter).
*   **Automatische Type Detectie:** Bepaalt het meest waarschijnlijke datatype per kolom (string, integer, number, boolean, empty).
*   **AI-Powered Beschrijvingen:** Gebruikt OpenAI (GPT-4o-mini) om zinvolle beschrijvingen voor API-parameters (kolommen) te genereren op basis van kolomnaam, voorbeelddata en optionele context.
*   **Configureerbare API:**
    *   Selecteer welke kolommen individueel doorzoekbaar moeten zijn via API-parameters.
    *   Configureer een optionele multi-kolom zoekparameter (bijv. `?query=zoekterm`) die meerdere tekstkolommen doorzoekt.
*   **Dynamische API Endpoint:** Genereert een API endpoint gebaseerd op de naam van het geüploade CSV-bestand (bijv. `/api/data/klanten_bestand`).
*   **PDF Hosting:** Eenvoudig uploaden en hosten van PDF-bestanden met een directe lokale en (optioneel via Ngrok) publieke URL.
*   **Gebruiksvriendelijke Web Interface:** Bootstrap 5 interface voor uploaden, configureren en bekijken van resultaten.
*   **Klaar voor Deployment:** Inclusief `requirements.txt` en `Procfile` voor eenvoudigere deployment naar PaaS-platformen.

## Setup & Installatie

1.  **Clone de Repository:**
    ```bash
    git clone <url_van_jouw_github_repository>
    cd <naam_van_repository_map>
    ```

2.  **Maak een Virtual Environment:** (Aanbevolen)
    ```bash
    python -m venv venv
    # Op Windows:
    .\venv\Scripts\activate
    # Op macOS/Linux:
    source venv/bin/activate
    ```

3.  **Installeer Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    **Let op:** Als je problemen ondervindt met ontbrekende dependencies, zorg ervoor dat je alle benodigde packages installeert:
    ```bash
    pip install python-dotenv chardet openai werkzeug jinja2
    ```

4.  **Installeer en configureer Ngrok (nodig voor publieke toegang):**
    *   Download Ngrok van [ngrok.com/download](https://ngrok.com/download)
    *   Plaats het `ngrok.exe` bestand in de hoofdmap van het project (naast `app.py`)
    *   Maak een gratis account aan op [ngrok.com](https://ngrok.com) om een authtoken te krijgen
    *   Je hebt deze authtoken nodig wanneer je de publieke toegang wilt starten via de applicatie

5.  **Configureer Environment Variabelen:**
    *   Kopieer `.env.example` naar een nieuw bestand genaamd `.env`:
        ```bash
        # Op Windows:
        copy .env.example .env
        # Op macOS/Linux:
        cp .env.example .env
        ```
    *   **Open het `.env` bestand** en vul je eigen waarden in:
        *   `FLASK_SECRET_KEY`: Genereer een sterke, willekeurige string. Je kunt Python gebruiken:
          ```python
          python -c 'import secrets; print(secrets.token_hex(16))'
          ```
          Plak de gegenereerde string in het `.env` bestand.
        *   `OPENAI_API_KEY`: Vul je eigen OpenAI API key in (beginnend met `sk-...`). Als je geen OpenAI API key hebt of deze functionaliteit niet wilt gebruiken, kun je dit veld leeg laten. De applicatie zal dan werken zonder de AI-gegenereerde kolombeschrijvingen.

## Lokaal Draaien

1.  **Zorg dat je virtual environment geactiveerd is.**
2.  **Navigeer naar de juiste directory:**
    ```bash
    cd csv-pdf-api-converter
    ```
3.  **Start de Flask Development Server:**
    ```bash
    python app.py
    ```

    Als je een foutmelding krijgt over een ontbrekend bestand, zorg ervoor dat je in de juiste directory staat. Je kunt ook het volledige pad naar het bestand gebruiken:
    ```bash
    python /volledig/pad/naar/csv-pdf-api-converter/app.py
    ```
4.  Open je webbrowser en ga naar `http://127.0.0.1:5000` (of het adres dat in de terminal wordt getoond).

## Gebruik

1.  **Startpagina:**
    *   Selecteer het **File Type** (CSV of PDF).
    *   *Indien CSV:* Vul een korte **Dataset Context** in (helpt de AI met beschrijvingen). Dit veld is optioneel/uitgegrijsd voor PDF.
    *   Kies het te uploaden bestand via **Choose File**.
    *   Klik op **Upload and Process**.

2.  **CSV Flow:**
    *   **Analyze Pagina (`/analyze`):**
        *   Bekijk de gedetecteerde kolommen, types, voorbeelden en AI-gegenereerde beschrijvingen.
        *   **Selecteer** de kolommen die je als **individuele API-parameters** wilt gebruiken (aanvinken in de "Selecteer" kolom).
        *   Optioneel: Activeer **Multi-Column Keyword Search**, geef een **Parameter Name** (bijv. `query`), en selecteer de kolommen waar deze parameter op moet zoeken via de checkboxes.
        *   Klik op **Apply Configuration & Generate API**.
    *   **API Ready Pagina (`/generate`):**
        *   Bekijk de geconfigureerde parameters.
        *   Kopieer de **Local API Endpoint** URL voor lokaal gebruik. De URL bevat nu een 'slug' gebaseerd op de bestandsnaam (bijv. `/api/data/klanten`).
        *   Indien nodig voor externe diensten (zoals HALO): Klik op **Start Public API (via Ngrok)**. Vul je Ngrok Authtoken in als daarom wordt gevraagd. De pagina zal herladen met de **Public API Endpoint** URL. Deze URL is tijdelijk.

3.  **PDF Flow:**
    *   **PDF Ready Pagina (`/generate_pdf`):**
        *   De pagina toont de **Local PDF Link**.
        *   Indien nodig voor externe diensten: Klik op **Start Public Access (via Ngrok)**. Vul je Ngrok Authtoken in. De pagina herlaadt met de **Public PDF Link**. Deze link is tijdelijk.

4.  **Start Over:** Klik op "Start Over / Upload New" om terug te gaan naar de startpagina en eventuele data/state te wissen (Ngrok blijft actief indien gestart).

## Deployment (Productie)

Deze applicatie lokaal draaien met `python app.py` en Ngrok is prima voor testen en ontwikkeling, maar **niet voor productie**.

*   **Gebruik Gunicorn:** De `Procfile` is inbegrepen om de app met Gunicorn te starten op platforms zoals Heroku of Render: `web: gunicorn app:app --timeout 120`.
*   **Platform Keuze:** Deploy naar een PaaS (Render, Heroku, PythonAnywhere) of via Docker naar een cloud provider (Google Cloud Run, AWS Fargate, etc.).
*   **Environment Variables:** Stel `FLASK_SECRET_KEY` en `OPENAI_API_KEY` in als environment variables op het hosting platform. Zet `FLASK_ENV=production`.
*   **Statische Bestanden & Uploads:**
    *   De `static` map wordt door Flask geserveerd, maar in productie is het efficiënter als een webserver (Nginx) dit doet (vaak geregeld door PaaS).
    *   De `uploads` map is **tijdelijk** op veel platforms. Als geüploade PDF's permanent bewaard moeten blijven, gebruik dan externe object storage (AWS S3, Google Cloud Storage, etc.) in plaats van de lokale `uploads` map. Pas de `/upload` en `/uploads/<filename>` routes dienovereenkomstig aan.
*   **Verwijder Ngrok Code:** De Ngrok-functionaliteit (`/start_ngrok`, etc.) is niet nodig en moet verwijderd of uitgeschakeld worden in een productie-deployment. De publieke URL wordt geleverd door het hosting platform.

## Configuratie

De volgende environment variabelen worden gebruikt (zie `.env.example`):

*   `FLASK_SECRET_KEY`: Essentieel voor sessiebeveiliging. **Moet geheim blijven.**
*   `OPENAI_API_KEY`: Je API key van OpenAI. **Moet geheim blijven.**
*   `FLASK_APP` (optioneel): Standaard `app.py`.
*   `FLASK_ENV` (optioneel): `development` of `production`.

## Dependencies

Zie `requirements.txt` voor de volledige lijst van dependencies. De belangrijkste zijn:

* pandas==2.2.1
* numpy==1.26.4
* flask==2.3.2
* requests==2.31.0
* python-dotenv==1.0.1
* chardet==5.2.0
* openai==1.70.0
* werkzeug==3.1.3
* jinja2==3.1.6

## Troubleshooting

### Ontbrekende Dependencies

Als je foutmeldingen krijgt over ontbrekende modules, installeer dan de ontbrekende dependencies:

```bash
pip install python-dotenv chardet openai werkzeug jinja2
```

### Problemen met het starten van de applicatie

1. **Foutmelding "No such file or directory"**: Zorg ervoor dat je in de juiste directory staat wanneer je `python app.py` uitvoert. Gebruik het volledige pad indien nodig.

2. **Foutmelding over ontbrekende .env file**: Kopieer het `.env.example` bestand naar `.env` en vul de benodigde waarden in.

3. **LLM features disabled waarschuwing**: Dit is normaal als je geen OpenAI API key hebt ingesteld. De applicatie werkt nog steeds, maar zonder AI-gegenereerde kolombeschrijvingen.

4. **Problemen met Ngrok**:
   - **Ngrok niet gevonden**: Zorg ervoor dat `ngrok.exe` in de hoofdmap van het project staat (naast `app.py`). Download het van [ngrok.com/download](https://ngrok.com/download) als je het nog niet hebt.
   - **Ongeldige authtoken**: Zorg ervoor dat je een geldige Ngrok authtoken hebt. Je kunt een gratis account aanmaken op [ngrok.com](https://ngrok.com) om een token te krijgen.
   - **Ngrok start niet**: Controleer of poort 5000 niet al in gebruik is door een andere applicatie. Probeer de applicatie te herstarten.
   - **Ngrok tunnel niet gevonden**: Soms duurt het even voordat de tunnel is opgezet. Wacht een paar seconden en probeer het opnieuw.
