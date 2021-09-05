mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"allana.silva@ihm.com.br\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml