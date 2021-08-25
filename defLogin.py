import streamlit as st
import hashlib
import sqlite3


# Def para Login

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# DB  Functions


def create_usertable(c, conn):
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username, password, c, conn):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',
              (username, password))
    conn.commit()


def login_user(username, password, c, conn):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',
              (username, password))
    data = c.fetchall()
    return data


def view_all_users(c, conn):
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data


def login(c, conn):
    st.sidebar.title('IHM Pandora Box')
    expanderLogin = st.sidebar.expander("Login", expanded=True)
    username = expanderLogin.text_input("User Name")
    password = expanderLogin.text_input("Password", type='password')
    check = expanderLogin.checkbox("Login")
    if check:
        hashed_pswd = make_hashes(password)
        result = login_user(username, check_hashes(
            password, hashed_pswd), c, conn)
        return result, username, check
    else:
        result = False
        return result, username, check
