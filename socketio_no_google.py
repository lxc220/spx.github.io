from flask import Flask, send_file, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, join_room, leave_room
import pyttsx3
import speech_recognition as sr
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import random
import deepl
from io import BytesIO
import uuid
import asyncio
import base64
from pydub import AudioSegment
import time
from opencc import OpenCC

#from recognition import settings, wav_to_png, recognition
NAME = "王聰賢"

HOST = '0.0.0.0'
PORT = 12345
BUFFER_SIZE = 4096


# ChatGPT API
load_dotenv(dotenv_path='.env')
client = AsyncOpenAI(api_key=os.getenv('CHATGPT_API_KEY'))

# 聲音路徑、TTS初始化
path = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_"        
engine = pyttsx3.init()

# gif、語言設定
J_gif_files = ['日文_女', '日文_女2', '日文_女3']
E_gif_files = ['英文_男', '英文_男2', '英文_男3']
C_gif_files = ['中文_女', '中文_女2', '中文_女3']

languages = [
    ('zh-TW', 'Chinese (traditional)', "中文"),
    ('en-US', 'English', "英文"),
    ('ja-JP', 'Japanese', "日文")
]

language_dict = {}

# 設定語言並加入字典
def language_settings():

    language_dict['Chinese (traditional)'] = {
        'greeting': '你好',
        'goodbye': '拜拜',
    }
    language_dict['English'] = {
        'greeting': 'Hello',
        'goodbye': 'bye-bye',
    }
    language_dict['Japanese'] = {
        'greeting': 'こんにちは',
        'goodbye': 'バイバイ',
    }



    

app = Flask(__name__, static_url_path='/static', static_folder='static')
socketio = SocketIO(app)

audio_dir = 'static/audio'
os.makedirs(audio_dir, exist_ok=True)

# 簡體字轉繁體字(包含慣用詞)
cc = OpenCC('s2twp')

# 在全局範圍內管理每個客戶端的狀態
client_states = {}

@app.route('/')
def index():
    return render_template('socketio.html')

@app.route('/gif/<path:filename>')
def serve_gif(filename):
    print(f"Serving GIF: {filename}")  # 除錯輸出
    return send_from_directory('static/gif', filename)

@app.route("/favicon.ico")
def favicon():
    return url_for('static', filename='data:,')

@app.route('/output.wav')
def get_output_audio():
    return send_from_directory('.', 'output.wav')

@socketio.on("connect")
def connect():
    # 取得客戶端的IP
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    print(f'\n\nClient connected from IP:{client_ip}\n\n')
    
    # 確認客戶端是否來自手機
    user_agent = request.headers.get('User-Agent')
    print(f"\n\n{user_agent}\n\n")
    is_mobile = 'Windows' not in user_agent and 'python' not in user_agent
    
    if is_mobile:
        # 手機設備使用 IP 作為 client_id
        client_id = client_ip
        join_room(client_ip)
    else:
        # 生成 UUID 作為非手機設備的 client_id
        client_id = str(uuid.uuid4())
        print(f"{client_id}")
        join_room(client_id)

    # 每位用戶的設定儲存
    client_states[client_id] = {
        'is_mobile': is_mobile,
        'transcript': None,
        'selected_lang': None,
        'selected_language': 0,
        'selected_send_lang': None,
        'selected_gif': None,
        'user_name': None,
        'c_name': None,
        'messages': None
    }

    # 回傳客戶端ID
    socketio.emit('ip', {'ip': client_id})

# 接收音檔
@socketio.on('send_audio')
def handle_audio(data):
    # 取得客戶端的IP
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    #client_ip = '192.168.0.11'
    # 取得傳送的文字
    try:
        # 取得音檔的 base64 編碼
        audio_base64 = data.get('audio')
        if audio_base64 is None:
            print('No audio data received.')
            return
        
        # 解碼 base64 編碼
        audio_data = base64.b64decode(audio_base64)
        
        # 將音檔保存為 WAV 文件
        with open('sio_from_client.wav', 'wb') as audio_file:
            audio_file.write(audio_data)
        
        print('Audio file saved as sio_from_client.wav')

    except Exception as e:
        print(f'Error processing audio data: {e}')
    
    type, message = handle_mobile_audio_input(audio_data, client_ip)
    state = client_states.get(client_ip, {})

    if message != 'error':
        if type == 'chat':
            if message == 'bye':
                socketio.emit('bye', room=client_ip)
                return
            ai_message = asyncio.run(mobile_chat(message, client_ip))
            if ai_message != 'error':
                socketio.emit('tts_start', room=client_ip)
                audio_data = mobile_tts(ai_message, state['selected_language'])
                text = ai_message
        elif type == 'greeting':
            socketio.emit('tts_start', room=client_ip)
            audio_data = mobile_tts(message, state['selected_language'])
            text = message
        # 獲取音頻時長
        duration = get_audio_duration(audio_data)
        print(f"Audio duration: {duration} seconds")
        print(f"Send lang: {state['selected_send_lang']}")
        # 利用socketio傳送給client
        socketio.emit('send_text', {'text':text, 'language':state['selected_send_lang']}, room=client_ip)

# 停止gif
@socketio.on('end')
def end_gif():
    # 取得客戶端的IP
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    print('\n\nGIF_END\n\n')
    socketio.emit('tts_finished', room=client_ip)

# 接收文字
@socketio.on('send_text')
def handle_text(data):
    # 取得客戶端的IP
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    #client_ip = '192.168.0.11'
    # 取得傳送的文字
    text_data = data.get('text')
    
    # 將文字保存為文本文件
    with open('text_from_client.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(text_data)
    
    print(f'Text saved as text_from_client.txt: {text_data}')

    type, message = handle_mobile_input(text_data, client_ip)
    state = client_states.get(client_ip, {})

    if message != 'error':
        if type == 'chat':
            if message == 'bye':
                socketio.emit('bye', room=client_ip)
                return
            ai_message = asyncio.run(mobile_chat(text_data, client_ip))
            if ai_message != 'error':
                socketio.emit('tts_start', room=client_ip)
                audio_data = mobile_tts(ai_message, state['selected_language'])
                text = ai_message
        elif type == 'greeting':
            socketio.emit('tts_start', room=client_ip)
            audio_data = mobile_tts(text_data, state['selected_language'])
            text = text_data
        # 獲取音頻時長
        duration = get_audio_duration(audio_data)
        print(f"Audio duration: {duration} seconds")

        # 利用socketio傳送給client
        socketio.emit('send_text', {'text':text}, room=client_ip)
        # 控制gif啟動、停止
        socketio.emit('tts_start', room=client_ip)
        time.sleep(duration)
        socketio.emit('tts_finished', room=client_ip)

def handle_mobile_input(text, client_ip):
    state = client_states.get(client_ip, {})

    if state['selected_language'] == 0:
        for lang, lang_dict_key, send_lang in languages:
            try:
                if text.lower() == language_dict[lang_dict_key]['greeting'].lower():
                    if lang_dict_key == 'Chinese (traditional)':
                        selected_gif = random.choice(C_gif_files)
                    elif lang_dict_key == 'English':
                        selected_gif = random.choice(E_gif_files)
                    elif lang_dict_key == 'Japanese':
                        selected_gif = random.choice(J_gif_files)
                    selected_lang = lang
                    selected_language = lang_dict_key
                    selected_send_lang = send_lang
                    # 設置狀態
                    state['selected_lang'] = selected_lang
                    state['selected_language'] = selected_language
                    state['selected_send_lang'] = selected_send_lang
                    state['selected_gif'] = selected_gif
                    
                    state['user_name'] = NAME
                    state['c_name'] = NAME

                    state['messages'] = [{"role": "system", "content": f"你是一位會聊天的老師，有各式各樣的興趣，也能夠糾正文法錯誤(不包含標點符號)。回答請在三句內(都用{selected_language}進行回答)"}]
                    socketio.emit('mobile_greeting', {'status': 'success', 'transcript': text, 'language': state['selected_language'], 'type': 'greeting', 'gif': state['selected_gif'], 'user_name': state['user_name'], 'c_name': state['c_name']}, room=client_ip)
                    return 'greeting', 'ok'
            except sr.UnknownValueError:
                print("==========error==========")
                continue
        socketio.emit('show', {'message': '請說出正確關鍵字'}, room=client_ip)
        return 'greeting', 'error'
        
    else:
        try:
            language=state['selected_language']
            
            goodbye_text = language_dict[language]['goodbye']
            if goodbye_text in text:
                return 'chat', 'bye'
            else:
                return 'chat', 'ok'
        except sr.UnknownValueError:
            print(f"==========error=========={state['selected_language']}")
            socketio.emit('show', {'message': 'Try Again'}, room=client_ip)
            return 'chat', 'error'

def handle_mobile_audio_input(audio_data, client_ip):
    state = client_states.get(client_ip, {})

    r = sr.Recognizer()
    # 使用 BytesIO 將 audio_data 轉換為類文件對象
    audio_file = BytesIO(audio_data)

    # 讀取音頻數據
    with sr.AudioFile(audio_file) as source:
        r.adjust_for_ambient_noise(source)
        audio = r.record(source)

    if state['selected_language'] == 0:
        for lang, lang_dict_key , send_lang in languages:
            try:
                text = r.recognize_google(audio, language=lang)
                print(f"greeting=========={text}===========")
                if text.lower() == language_dict[lang_dict_key]['greeting'].lower():
                    if lang_dict_key == 'Chinese (traditional)':
                        selected_gif = random.choice(C_gif_files)
                    elif lang_dict_key == 'English':
                        selected_gif = random.choice(E_gif_files)
                    elif lang_dict_key == 'Japanese':
                        selected_gif = random.choice(J_gif_files)
                    selected_lang = lang
                    selected_language = lang_dict_key
                    selected_send_lang = send_lang
                    # 設置狀態
                    state['selected_lang'] = selected_lang
                    state['selected_language'] = selected_language
                    state['selected_send_lang'] = selected_send_lang
                    state['selected_gif'] = selected_gif
                    
                    state['user_name'] = NAME
                    state['c_name'] = NAME

                    state['messages'] = [{"role": "system", "content": f"你是一位會聊天的老師，有各式各樣的興趣，也能夠糾正文法錯誤(不包含標點符號)。回答請在兩句內(都用{selected_language}進行回答)"}]
                    socketio.emit('mobile_greeting', {'status': 'success', 'transcript': text, 'language': state['selected_language'], 'type': 'greeting', 'gif': state['selected_gif'], 'user_name': state['user_name'], 'c_name': state['c_name']}, room=client_ip)
                    return 'greeting', text
            except sr.UnknownValueError:
                print("==========error==========")
                continue
        socketio.emit('show', {'message': '請說出正確關鍵字'}, room=client_ip)
        return 'greeting', 'error'
        
    else:
        try:
            text = r.recognize_google(audio, language=state['selected_lang'])
            print(f"chat=========={text}==========")
            language=state['selected_language']
            
            goodbye_text = language_dict[language]['goodbye']
            if goodbye_text in text:
                return 'chat', 'bye'
            else:
                return 'chat', text
        except sr.UnknownValueError:
            print(f"==========error=========={state['selected_language']}")
            socketio.emit('show', {'message': 'Try Again'}, room=client_ip)
            return 'chat', 'error'
                
@app.route('/handle_input', methods=['POST'])
def handle_input():
    client_id = request.args.get('client_id')
    state = client_states.get(client_id, {})
    print(f"\n\nID:{client_states[client_id]}\n\n")
    
    # 獲取音頻
    audio_file = request.files['audio']

    # 將接收到的音頻文件讀取到內存中
    audio_data = BytesIO(audio_file.read())

    # 將音頻轉換為 WAV 格式
    try:
        audio = AudioSegment.from_file(audio_data)  # 讀取音頻數據
        wav_data = BytesIO()
        audio.export(wav_data, format='wav')  # 將音頻數據導出為 WAV 格式
        wav_data.seek(0)  # 將指針重置到文件開頭
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error converting audio: {str(e)}'})

    r = sr.Recognizer()
    # 音頻轉文字
    with sr.AudioFile(wav_data) as source:
        audio = r.record(source)

    if state['selected_language'] == 0:
        
        for lang, lang_dict_key, send_lang in languages:
            try:
                text = r.recognize_google(audio, language=lang)
                print(f"greeting=========={text}===========")
                if text.lower() == language_dict[lang_dict_key]['greeting'].lower():
                    if lang_dict_key == 'Chinese (traditional)':
                        selected_gif = random.choice(C_gif_files)
                    elif lang_dict_key == 'English':
                        selected_gif = random.choice(E_gif_files)
                    elif lang_dict_key == 'Japanese':
                        selected_gif = random.choice(J_gif_files)
                    selected_lang = lang
                    selected_language = lang_dict_key
                    selected_send_lang = send_lang
                    
                    # 設置狀態
                    state['selected_lang'] = selected_lang
                    state['selected_language'] = selected_language
                    state['selected_send_lang'] = selected_send_lang
                    state['selected_gif'] = selected_gif
                    state['messages'] = [{"role": "system", "content": f"你是一位會聊天的老師，能夠糾正用字錯誤(不包含標點符號)。回答請在三句內(都用{selected_language}進行回答)"}]
                    # 身份辨識
                    """wav_to_png()
                    c_name, e_name, j_name = recognition()
                    if selected_language == 'English':
                        user_name = e_name
                    elif selected_language == 'Japanese':
                        user_name = j_name
                    else:
                        user_name = c_name
                    state['user_name'] = user_name"""
                    state['user_name'] = NAME
                    state['c_name'] = NAME
                    
                    client_states[client_id] = state

                    return jsonify({'status': 'success', 'transcript': text, 'language': state['selected_language'], 'type': 'greeting', 'gif': state['selected_gif'], 'user_name': state['user_name'], 'c_name': state['c_name']})
            except sr.UnknownValueError:
                print("==========error==========")
                continue
        return jsonify({'status': 'error', 'message': '請說出正確關鍵字'})
        #socketio.emit('show', {'message': '請說出正確關鍵字'}, room=client_id)
    else:
        try:
            text = r.recognize_google(audio, language=state['selected_lang'])
            
            print(f"chat=========={text}==========")
            language=state['selected_language']
            
            goodbye_text = language_dict[language]['goodbye']
            if goodbye_text in text:
                return jsonify({'status': 'goodbye', 'transcript': text, 'language': state['selected_language']})
            return jsonify({'status': 'success', 'transcript': text, 'language': state['selected_language'], 'type': 'chat', 'gif': state['selected_gif'], 'user_name': state['user_name'], 'c_name': state['c_name']})
        except sr.UnknownValueError:
            print(f"==========error=========={state['selected_language']}")
            return jsonify({'status': 'error', 'message': 'Try Again'})

async def mobile_chat(user_message, client_ip):
    state = client_states.get(client_ip, {})
    print(f"Client ID: {client_ip}")
    print(f"Client State: {state}")

    socketio.emit('show', {'message': 'Loading...'}, room=client_ip)
    try:
        client_states[client_ip]['messages'].append({"role": "user", "content": user_message})
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=client_states[client_ip]['messages']
        )
        ai_response = response.choices[0].message.content
        client_states[client_ip]['messages'].append({"role": "system", "content": ai_response})
        print(ai_response)
        ai_translate = translate(ai_response)

        socketio.emit('mobile_chat', {'userInput': user_message, 'response': ai_response, 'language': state['selected_language'], 'translate': ai_translate}, room=client_ip)
        return ai_response
    
    except Exception as e:
        return 'error'
    
@app.route('/chat', methods=['POST'])
async def chat():
    client_id = request.args.get('client_id')  # 從查詢參數中獲取 client_id
    state = client_states.get(client_id, {})
    print(f"Client ID: {client_id}")
    print(f"Client State: {state}")

    try:
        data = request.get_json()
        user_message = data['message']
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        client_states[client_id]['messages'].append({"role": "user", "content": user_message})
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=client_states[client_id]['messages']
        )
        ai_response = response.choices[0].message.content
        client_states[client_id]['messages'].append({"role": "system", "content": ai_response})
        print(ai_response)
        ai_translate = translate(ai_response)
        
        return jsonify({"response": ai_response, "language": state.get('selected_language'), "translate": ai_translate})
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/tts', methods=['POST'])
def text_to_speech():
    client_id = request.args.get('client_id')  # 從查詢參數中獲取 client_id
    state = client_states.get(client_id, {})
    data = request.get_json()
    text = data['text']
    language = data['language']
    # 前次engine必須先終止
    try:
        engine.endLoop()
    except:
        pass

    if language == 'Japanese':
        engine.setProperty('voice', path + "jaJP_AyumiM") # 女
        engine.setProperty('rate', 180) # 調整語速 
        
    elif language == 'English':
        engine.setProperty('voice', path + "enUS_DavidM") # 男
        engine.setProperty('rate', 160) # 調整語速
        
    elif language == 'Chinese (traditional)':
        engine.setProperty('voice', path + "zhTW_YatingM") # 女
        engine.setProperty('rate', 180) # 調整語速
        
    elif language == 'Vietnamese':        
        engine.setProperty('voice', path + "viVN_An") # 男    
        engine.setProperty('rate', 150) # 調整語速  
        
    filename = f'output.wav'
    filename = f'{str(uuid.uuid4())}.wav'
    file_path = os.path.join(audio_dir, filename)

    engine.save_to_file(text, file_path)
    engine.runAndWait()

    return jsonify({'status': 'success', 'audio_url': f'{filename}'})

def mobile_tts(text, language):
    # 前次engine必須先終止
    try:
        engine.endLoop()
    except:
        pass

    if language == 'Japanese':
        engine.setProperty('voice', path + "jaJP_AyumiM") # 女
        engine.setProperty('rate', 180) # 調整語速 
        
    elif language == 'English':
        engine.setProperty('voice', path + "enUS_DavidM") # 男
        engine.setProperty('rate', 160) # 調整語速
        
    elif language == 'Chinese (traditional)':
        engine.setProperty('voice', path + "zhTW_YatingM") # 女
        engine.setProperty('rate', 180) # 調整語速
        
    elif language == 'Vietnamese':        
        engine.setProperty('voice', path + "viVN_An") # 男    
        engine.setProperty('rate', 150) # 調整語速  

    # 將音頻保存到臨時文件
    temp_filename = 'temp_audio.wav'
    engine.save_to_file(text, temp_filename)
    engine.runAndWait()

    # 讀取臨時文件的內容到 BytesIO 中
    audio_data = BytesIO()
    with open(temp_filename, 'rb') as f:
        audio_data.write(f.read())

    # 確保指針回到文件開頭
    audio_data.seek(0)

    # 刪除臨時文件
    os.remove(temp_filename)

    # 將音頻數據轉換為 base64
    audio_base64 = base64.b64encode(audio_data.getvalue()).decode('utf-8')

    # 通過 Socket.IO 發送音頻數據給客戶端
    #socketio.emit('receive_audio', {'audio': audio_base64})

    # 返回音頻數據
    return audio_data.getvalue()

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory(audio_dir, filename)

@app.route('/delete_audio', methods=['DELETE'])
def delete_audio():
    filename = request.args.get('filename')
    try:
        os.remove(os.path.join('static/audio', filename))
        return jsonify({'status': 'success'})
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

# deepl API
load_dotenv(dotenv_path='.env')
auth_key = os.getenv(key='DEEPL_API_KEY')
deepl_translator = deepl.Translator(auth_key)

# deepl翻譯
def translate(text):
    result = deepl_translator.translate_text(text, target_lang="ZH").text
    print("\ndeepl:" + result)
    tw_result = cc.convert(result)
    print("OpenCC" + tw_result + "\n")
    return tw_result

def get_audio_duration(audio_data):
    audio = AudioSegment.from_file(BytesIO(audio_data), format="wav")
    return audio.duration_seconds
if __name__ == '__main__':
    language_settings()
    socketio.run(app, '0.0.0.0', debug=True, port=5000)
    #socketio.run(app, '0.0.0.0', debug=True, port=5000, ssl_context=('server.crt', 'server.key'))
