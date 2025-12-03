import streamlit as st
import time

def stream_text(text):
    for char in text:
        yield char
        time.sleep(0.02)

def user_text(input_text):
    files_info = []
    
    if input_text.files:
        for f in input_text.files:
            files_info.append({"name": f.name, "type": f.type, "data": f.read()})

    st.session_state["messages"].append({"role": "user", "content": input_text.text, "avatar": "assets/thunderatz.png", "files": files_info})

    user = st.chat_message("user", avatar="assets/thunderatz.png")
    
    for f in files_info:
        if f["type"].startswith("image/"):
            user.image(f["data"])
        elif f["type"] == "application/pdf":
            user.download_button(f["name"], f["data"], f["name"])
        else:
            user.write(f"Arquivo enviado: {f['name']}")

    user.write(input_text.text)

def ia_response(response):
    st.session_state["messages"].append({"role": "assistant", "content": response, "avatar": "ai"})
    ai = st.chat_message("assistant", avatar="ai")
    ai.write_stream(stream_text(response))

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    chat = st.chat_message(msg["role"], avatar=msg.get("avatar"))
    chat.write(msg["content"])

    if "files" in msg:
        for f in msg["files"]:
            if f["type"].startswith("image/"):
                chat.image(f["data"])
            elif f["type"] == "application/pdf":
                chat.download_button(f["name"], f["data"], f["name"])
            else:
                chat.write(f"Arquivo enviado: {f['name']}")


input_box = st.chat_input(placeholder="Your message", key=None, max_chars=None, accept_file="multiple", file_type=None, disabled=False, on_submit=None, args=None, kwargs=None, width="stretch")

if input_box:

    user_text(input_box)

    time.sleep(1)
    
    response = "Thunderatz <3<3<3"
    ia_response(response)    