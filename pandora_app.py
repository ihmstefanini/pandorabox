"""Main module for the Pandor Box app"""
import streamlit as st
import awesome_streamlit as ast
import hashlib
import sqlite3

# Importa as bibliotecas criadas em arquivo .py
import defSessionState as ss
import defLogin as lg

# Importa os arquivos em python que representam cada página do app
import home
import dataPreparation
import dataUnderstanding
import dataSync
#import about

st.set_page_config(
    # Can be "centered" or "wide". In the future also "dashboard", etc.
    layout="centered",
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    # String or None. Strings get appended with "• Streamlit".
    page_title="IHM - Pandora",
    page_icon=None,  # String, anything supported by st.image, or None.
)

ast.core.services.other.set_logging_format()

conn = sqlite3.connect('data.db')
c = conn.cursor()

PAGES = {
    "Home": home,
    "Data Preparation": dataPreparation,
    "Data Correlation": dataUnderstanding,
    "Data Syncronization": dataSync,
}


def main():
    """Main function of the App"""

    state = ss._get_state()

    result, username, check = lg.login(c, conn)

    if result and check:

        lg.createUser(username, c, conn)

        st.sidebar.success("Logged In as {}".format(username))
        st.sidebar.title("Navigation")
        selection = st.sidebar.radio("Go to", list(PAGES.keys()))

        page = PAGES[selection].write(state)

        st.sidebar.title("About")
        st.sidebar.info(
            """
            Part of this app is maintained by IHM Stefanini. You can learn more about us at
            [IHM Stefanini](https://ihm.com.br)
            """
        )

    elif not result and check:
        st.sidebar.warning("Incorrect Username/Password")

    state.sync()


if __name__ == "__main__":
    main()
