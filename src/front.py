import streamlit as st
import time

def stream_text(text):
    for char in text:
        yield char
        time.sleep(0.02)

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


text_input = st.chat_input(placeholder="Your message", key=None, max_chars=None, accept_file="multiple", file_type=None, disabled=False, on_submit=None, args=None, kwargs=None, width="stretch")

if text_input:
    files_info = []

    if text_input.files:
        for f in text_input.files:
            files_info.append({"name": f.name, "type": f.type, "data": f.read()})

    st.session_state["messages"].append({"role": "user", "content": text_input.text, "avatar": "assets/thunderatz.png", "files": files_info})

    user = st.chat_message("user", avatar="assets/thunderatz.png")
    
    for f in files_info:
        if f["type"].startswith("image/"):
            user.image(f["data"])
        elif f["type"] == "application/pdf":
            user.download_button(f["name"], f["data"], f["name"])
        else:
            user.write(f"Arquivo enviado: {f['name']}")
    user.write(text_input.text)

    time.sleep(1)

    ai_response = "A THUNDERATZ <3<3<3"

    st.session_state["messages"].append({
        "role": "assistant",
        "content": ai_response,
        "avatar": "ai"
    })

    ai = st.chat_message("assistant", avatar="ai")
    ai.write_stream(stream_text(ai_response))