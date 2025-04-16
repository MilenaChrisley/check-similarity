from flask import Flask, request, render_template, jsonify
import video_quality
import threading
import os
import time  # Importa o módulo time

app = Flask(__name__)

engagement_data = {}

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    # Redefinir dados de análise quando a página inicial for carregada
    video_quality.analysis_data = {}
    video_quality.degradation_records = []
    return render_template('index.html')

@app.route('/load_video', methods=['POST'])
def load_video():
    video_source = request.form['video_source']
    if video_source == 'youtube':
        video_url = request.form['youtube_url']
        video_path = os.path.join(UPLOAD_FOLDER, 'downloaded_video.mp4')
        video_quality.download_youtube_video(video_url, video_path)
    else:
        video_file = request.files['video_file']
        video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        video_file.save(video_path)
    
    video_quality.analysis_data['video_path'] = video_path
    video_quality.analysis_data['analysis_complete'] = False
    video_quality.analysis_data['progress'] = 0  # Inicializar progresso

    def run_analysis(video_path):
        video_quality.analyze_video(video_path)

    analysis_thread = threading.Thread(target=run_analysis, args=(video_path,))
    analysis_thread.start()

    return render_template('load_video.html', video_path=os.path.basename(video_path))

@app.route('/update_engagement', methods=['POST'])
def update_engagement():
    data = request.get_json()
    watched_time = data['watched_time']
    total_duration = data['total_duration']
    video_path = data['video_path']
    engagement_data[video_path] = {
        'watched_time': watched_time,
        'total_duration': total_duration
    }
    print('Received engagement data:', engagement_data[video_path])  # Adicione este print para debug
    return jsonify({'status': 'success'}), 204

"""    return '', 204"""

@app.route('/get_analysis_status', methods=['GET'])
def get_analysis_status():
    return jsonify(video_quality.analysis_data)

@app.route('/result')
def result():
    return render_template('result.html', result=video_quality.analysis_data)

if __name__ == '__main__':
    app.run(debug=True)

