from flask import Flask, send_file, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, join_room, leave_room
import pyttsx3
import speech_recognition as sr
from googletrans import Translator
from googletrans import LANGUAGES
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from tenacity import retry, stop_after_attempt, wait_fixed
import random
import tempfile
import subprocess
import wave
import pygame
import deepl
import socket
import threading
from io import BytesIO
import uuid
import asyncio
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import load_model  # TensorFlow is required for Keras to work 把tensorflow降到2.12.0版本才能用
from PIL import Image
import numpy as np
from pydub import AudioSegment
import base64

s = None
is_mobile = False

HOST = '0.0.0.0'
PORT = 12345
BUFFER_SIZE = 1024

# ChatGPT API
load_dotenv(dotenv_path='.env')
client = AsyncOpenAI(api_key=os.getenv('CHATGPT_API_KEY'))
# 聲音路徑、TTS初始化
path = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_"        
engine = pyttsx3.init()

J_gif_files = ['日文_女', '日文_女2', '日文_女3']
E_gif_files = ['英文_男', '英文_男2', '英文_男3']
C_gif_files = ['中文_女', '中文_女2', '中文_女3']

language_list = [['Japanese', 30, 25, 25],
                 ['English', 30, 20, 20],
                 ['Chinese (traditional)', 15, 20, 20],
                 ['Vietnamese', 120, 130, 100]]

languages = [
    ('zh-TW', 'Chinese (traditional)'),
    ('en-US', 'English'),
    ('ja-JP', 'Japanese')
]

# 設定語言並加入字典
def language_settings(language_list):
    global language_dict
    
    language_dict = {}
    
    for language_info in language_list:
        language = language_info[0]
        language_key = language.lower()

        if language_key in LANGUAGES.values():
            dest = language_key
            translator = Translator()

            greeting_text = 'Hello'
            goodbye_text = 'bye-bye'
            translated_greeting = translator.translate(greeting_text, dest=dest).text
            translated_goodbye = translator.translate(goodbye_text, dest=dest).text

            language_dict[language] = {
                'greeting': translated_greeting,
                'goodbye': translated_goodbye,
                'gif1_delay': language_info[1],
                'gif2_delay': language_info[2],
                'gif3_delay': language_info[3]
            }
        else:
            print(f"{language_key.capitalize()} 不在字典中")

    language_dict['Chinese (traditional)']['goodbye'] = '拜拜'

app = Flask(__name__, static_url_path='/static', static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")

audio_dir = 'static/audio'
os.makedirs(audio_dir, exist_ok=True)

# 在全局範圍內管理每個客戶端的狀態
client_states = {}
client_id = '123456'
first_start = True
@app.route('/')
def index():
    global first_start
    user_agent = request.headers.get('User-Agent')
    is_mobile = 'Windows' not in user_agent
    if is_mobile:
        print("is mobile\n\n")
        client_id = 'phone'
        client_states[client_id] = {
            'is_mobile': is_mobile,
            'transcript': None,
            'selected_lang': None,
            'selected_language': 0,
            'selected_gif': None,
            'user_name': None,
            'messages': None
        }
        threading.Thread(target=main, args=(client_id, )).start()
    else:
        client_id = 'desktop'
        client_id = str(uuid.uuid4())  # 為每個客戶端創建唯一的房間ID
        client_states[client_id] = {
            'is_mobile': is_mobile,
            'transcript': None,
            'selected_lang': None,
            'selected_language': 0,
            'selected_gif': None,
            'user_name': None,
            'messages': None
        }
        
    #client_id = str(uuid.uuid4())  # 為每個客戶端創建唯一的房間ID
    
    
    # 第一次進入網頁開啟socket
    #if first_start:
    #    threading.Thread(target=main).start()
        #first_start = False

    return render_template('index.html', is_mobile=is_mobile, client_id=client_id)

@app.route('/gif/<path:filename>')
def serve_gif(filename):
    print(f"Serving GIF: {filename}")  # 除錯輸出
    return send_from_directory('static/gif', filename)

@app.route("/favicon.ico")
def favicon():
    return url_for('static', filename='data:,')

@socketio.on("connect")
def connect():
    print("client wants to connect")
    socketio.emit("status", { "data": "Connected. Hello!" })

@socketio.on("join")
def on_join(data):
    user = data["user"]
    client_id = data["room"]  # 從 data 中獲取 client_id
    room = client_id
    print(f"client {user} wants to join: {room}")
    join_room(room)
    socketio.emit("room_message", f"Welcome to {room}, {user}", room=room)

@socketio.on('send_audio')
def handle_audio(data):
    # 取得音檔的 base64 編碼
    audio_base64 = data.get('audio')
    # 解碼 base64 編碼
    audio_data = base64.b64decode(audio_base64)
    # 將音檔保存為 WAV 文件
    with open('received_audio.wav', 'wb') as audio_file:
        audio_file.write(audio_data)
    print('Audio file saved as received_audio.wav')


@app.route('/handle_input', methods=['POST'])
def handle_input():
    client_id = request.args.get('client_id')
    state = client_states.get(client_id, {})
    print(f'My ID:{client_id}')
    print(f'Selected Language: {state.get("selected_lang")}')
    if not state['is_mobile']:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            
            print("請說話...")
            socketio.emit('show', {'message': 'Say Something...'}, room=client_id)
            audio = r.listen(source, phrase_time_limit=5)
            print("錄音中...\n")
            socketio.emit('show', {'message': 'Loading...'}, room=client_id)
        
        # 將錄音資料轉換為 WAV 格式的資料流
        wav_io = BytesIO(audio.get_wav_data())

    if state['selected_language'] == 0:
        
        for lang, lang_dict_key in languages:
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
                    
                    # 設置狀態
                    state['selected_lang'] = selected_lang
                    state['selected_language'] = selected_language
                    state['selected_gif'] = selected_gif
                    state['messages'] = [{"role": "system", "content": f"你是一位且喜歡各式各樣東西的聊天專家，回答請在兩句內(都用{selected_language}進行回答)"}]
                    # 身份辨識
                    img_square = wav_to_png(wav_io)
                    c_name, e_name, j_name = recognition(img_square)
                    if selected_language == 'English':
                        user_name = e_name
                    elif selected_language == 'Japanese':
                        user_name = j_name
                    else:
                        user_name = c_name
                    state['user_name'] = user_name
                    
                    client_states[client_id] = state
                    return jsonify({'status': 'success', 'transcript': text, 'language': state['selected_language'], 'type': 'greeting', 'gif': state['selected_gif'], 'user_name': state['user_name'], 'c_name': c_name})
            except sr.UnknownValueError:
                print("==========error==========")
                continue
        
        socketio.emit('show', {'message': '請說出正確關鍵字'}, room=client_id)
    else:
        try:
            text = r.recognize_google(audio, language=state['selected_lang'])
            print(f"chat=========={text}==========")
            return jsonify({'status': 'success', 'transcript': text, 'language': state['selected_language'], 'type': 'chat', 'gif': state['selected_gif'], 'user_name': state['user_name']})
        except sr.UnknownValueError:
            print(f"==========error=========={state['selected_language']}")
            return jsonify({'status': 'error', 'message': 'Try Again'})
     # 如果以上條件都不符合，返回預設的錯誤響應
    return jsonify({'status': 'error', 'message': 'Unhandled case'})

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
            model="gpt-3.5-turbo",
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
        
    #engine.say(text)        
    #engine.runAndWait()
    filename = f'output.wav'
    file_path = os.path.join(audio_dir, filename)

    engine.save_to_file(text, file_path)
    engine.runAndWait()


    if not state['is_mobile']:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        socketio.emit('tts_start', room=client_id)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        socketio.emit('tts_finished', room=client_id)
        pygame.mixer.quit()
    """
    else:
        try:
            socketio.emit('tts_start', room=client_id)
            with open(filename, 'rb') as f:
                while (chunk := f.read(BUFFER_SIZE)):
                    s.sendall(chunk, addr)
                

            print('\n\n==========File sent successfully==========\n\n')
        except Exception as e:
            print(f"Failed to send file: {e}")
    """


    return jsonify({'status': 'success', 'audio_url': f'{filename}'})

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


google_translator = Translator()
# google翻譯
def google_translate(text):   
    # 轉為繁體中文
    result = google_translator.translate(text, dest='zh-TW').text
    print("google:" + result)
    #add_message('中文:' + result + '\n')
    return result

# deepl API
load_dotenv(dotenv_path='.env')
auth_key = os.getenv(key='DEEPL_API_KEY')
deepl_translator = deepl.Translator(auth_key)

# deepl翻譯
def translate(text):
    result = deepl_translator.translate_text(text, target_lang="ZH").text
    print("deepl:" + result)
    result = google_translate(result)
    return result

import time

@app.route('/start_socket', methods=['POST'])
async def socket_listener(client_id):
    #client_id = request.args.get('client_id')  # 從查詢參數中獲取 client_id
    state = client_states.get(client_id, {})
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 重整網頁會自動清除
    s.bind((HOST, PORT))
    s.listen(5)  # TCP listen for connections

    print('Socket server started at: %s:%s' % (HOST, PORT))
    print('\n\n====================Waiting for connection...====================\n\n')
    
    while True:
        conn, addr = s.accept()  
        received_data = bytearray()  # 用來儲存接收到的數據
        try:
            while True:
                # 接收數據
                cdata= conn.recv(BUFFER_SIZE)
                # 檢查是否是結束標記
                if cdata == b'END' or not cdata:          #轉為字節串（bytes）
                    print('File received successfully')
                    break
                # 寫入檔案
                received_data.extend(cdata)
            filename = '123.wav'
            with open(filename, 'wb') as f:
                f.write(received_data)
            print(f"Audio file saved as {filename}")
            type, message = handle(received_data, client_id)
            if message != 'error':
                if type == 'chat':
                    ai_message = await mobile_chat(message, client_id)
                    if ai_message != 'error':
                        # 回傳文字
                        conn.send(ai_message.encode())
                        socketio.emit('tts_start', room=client_id)
                        # 回傳音檔
                        """
                        audio_data = mobile_tts(ai_message, state['selected_language'])
                        # 獲取音頻時長
                        duration = get_audio_duration(audio_data)
                        print(f"Audio duration: {duration} seconds")
                        #conn.sendall(audio_data)
                        index = 0
                        while index < len(audio_data):
                            conn.sendall(audio_data[index:index + BUFFER_SIZE])
                            index += BUFFER_SIZE
                        conn.sendall(b'END')
                        socketio.emit('tts_start', room=client_id)
                        time.sleep(duration)
                        socketio.emit('tts_finished', room=client_id)
                        """
                    else:
                        conn.send(b"NO")
                elif type == 'greeting':
                    # 回傳文字
                    conn.send(message.encode())
                    socketio.emit('tts_start', room=client_id)
                    # 回傳音檔
                    """
                    audio_data = mobile_tts(message, state['selected_language'])
                    # 獲取音頻時長
                    duration = get_audio_duration(audio_data)
                    print(f"Audio duration: {duration} seconds")
                    #conn.sendall(audio_data)
                    index = 0
                    while index < len(audio_data):
                        conn.sendall(audio_data[index:index + BUFFER_SIZE])
                        index += BUFFER_SIZE
                    conn.sendall(b'END')
                    socketio.emit('tts_start', room=client_id)
                    time.sleep(duration)
                    socketio.emit('tts_finished', room=client_id)
                    """
            else: 
                # 回傳 NO
                conn.send(b"NO")
            
                    
        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            conn.close()

def handle(received_data, client_id):
    state = client_states.get(client_id, {})
    # 使用 BytesIO 將 received_data 轉換為類文件對象
    audio_file = BytesIO(received_data)

    r = sr.Recognizer()

    # 讀取音頻數據
    with sr.AudioFile(audio_file) as source:
        r.adjust_for_ambient_noise(source)
        audio = r.record(source)
    
    if state['selected_language'] == 0:
        for lang, lang_dict_key in languages:
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
                    # 設置狀態
                    state['selected_lang'] = selected_lang
                    state['selected_language'] = selected_language
                    state['selected_gif'] = selected_gif
                    state['messages'] = [{"role": "system", "content": f"你是一位且喜歡各式各樣東西的聊天專家，回答請在兩句內(都用{selected_language}進行回答)"}]
                    
                    #身分辨識
                    wav_io = BytesIO(audio.get_wav_data())  # 將錄音資料轉換為 WAV 格式的資料流
                    img_square = wav_to_png(wav_io)     # 將 wav_io 傳遞給 wav_to_png 函數進行處理
                    c_name, e_name, j_name = recognition(img_square)
                    if state['selected_language'] == 'English':
                        user_name = e_name
                    elif state['selected_language'] == 'Japanese':
                        user_name = j_name
                    else:
                        user_name = c_name
                    state['user_name'] = user_name

                    socketio.emit('mobile_greeting', {'status': 'success', 'transcript': text, 'language': state['selected_language'], 'type': 'greeting', 'gif': state['selected_gif'], 'user_name': state['user_name'], 'c_name': c_name}, room=client_id)
                    return 'greeting', text
            except sr.UnknownValueError:
                print("==========error==========")
                continue
        socketio.emit('show', {'message': '請說出正確關鍵字'}, room=client_id)
        return 'greeting', 'error'
        
    else:
        try:
            text = r.recognize_google(audio, language=state['selected_lang'])
            print(f"chat=========={text}==========")
            
            return 'chat', text
        except sr.UnknownValueError:
            print(f"==========error=========={state['selected_language']}")
            socketio.emit('show', {'message': 'Try Again'}, room=client_id)
            return 'chat', 'error'

async def mobile_chat(user_message, client_id):
    #client_id = request.args.get('client_id')  # 從查詢參數中獲取 client_id
    state = client_states.get(client_id, {})
    print(f"Client ID: {client_id}")
    print(f"Client State: {state}")

    socketio.emit('show', {'message': 'Loading...'}, room=client_id)
    try:
        client_states[client_id]['messages'].append({"role": "user", "content": user_message})
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=client_states[client_id]['messages']
        )
        ai_response = response.choices[0].message.content
        client_states[client_id]['messages'].append({"role": "system", "content": ai_response})
        print(ai_response)
        ai_translate = translate(ai_response)

        socketio.emit('mobile_chat', {'userInput': user_message, 'response': ai_response, 'language': state['selected_language'], 'translate': ai_translate}, room=client_id)
        return ai_response
    
    except Exception as e:
        return 'error'

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

    # 返回音頻數據
    return audio_data.getvalue()

def get_audio_duration(audio_data):
    audio = AudioSegment.from_file(BytesIO(audio_data), format="wav")
    return audio.duration_seconds

# 在主程序中啟動異步環境
def main(client_id):
    asyncio.run(socket_listener(client_id))

def re_settings():
    global sample_rate, model, class_names, data
    # 採樣率
    sample_rate = 11025

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    
    # Load the model
    # 新參數，XY軸，降躁，採樣率:11025
    model = load_model("keras_model(librosa_11025).h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r", encoding='UTF-8').readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
def wav_to_png(wav_io):
    # 讀取音訊
    audio, sr = librosa.load(wav_io, sr=sample_rate) # y=音頻速率 sr=採樣率

    # 降躁處理
    y = librosa.effects.trim(audio, top_db=20, frame_length=2048, hop_length=512)[0]    # top_db:低於20dB靜音
    #y = nr.reduce_noise(y=y, sr=sr)

    # 生成頻譜圖
    D = librosa.stft(y, n_fft=1024, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect')
    S_db = librosa.amplitude_to_db(abs(D))
    
    # 繪製並保存頻譜圖
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(S_db, sr=sr, cmap='inferno', x_axis='time', y_axis='hz')
    
    plt.tight_layout()

    # 將圖像儲存在 BytesIO 物件中
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()
    
    # 讀取圖像並調整大小為224x224
    img = Image.open(img_io).convert("RGB")  # 確保圖像為RGB格式
    img_square = img.resize((224, 224)) 
    return img_square
    


def recognition(img_square):   
    # turn the image into a numpy array
    image_array = np.asarray(img_square)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    # Print prediction and confidence score
    parts = class_name.split()
    # 英文部分在第二個元素
    e_name = parts[1]
    c_name = parts[2]
    j_name = parts[3]
    
    print("You are:", class_name[2:], end="")
    print(f"英文:{e_name}\n中文:{c_name}\n日文:{j_name}")
    print("Confidence Score:", confidence_score*100, "%")
    return c_name, e_name, j_name

def record():
    # 創建一個語音識別器物件
    r = sr.Recognizer()
    
    # 使用系統默認的麥克風錄製聲音
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("初始化...")
        audio_data = r.record(source, duration=1)  # 錄製5秒聲音
        #audio_data = r.listen(source, phrase_time_limit=5)
    
    # 將錄音資料轉換為 WAV 格式的資料流
    wav_io = BytesIO(audio_data.get_wav_data())
    print("初始化完成")
    return wav_io

def init():
    wav_io = record()
    img_square = wav_to_png(wav_io)
    recognition(img_square)

if __name__ == '__main__':
    language_settings(language_list)    # 語言設定
    re_settings()                       # 身分辨識設定
    init()                              # 身分辨識初始化
    socketio.run(app, '0.0.0.0', debug=True, port=5000)
    #socketio.run(app, '0.0.0.0', debug=True, port=5000, ssl_context=('server.crt', 'server.key'))
