from flask import Flask, Blueprint, request, jsonify
from flask_restx import Resource, Api, Namespace, reqparse
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.datastructures import FileStorage
from datetime  import date, datetime, timedelta
import jwt
from json import dumps
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import base64
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import os
import imghdr
import re

from chatbot import *
from transformers import BertTokenizer, TFBertForSequenceClassification
import random

from dotenv import load_dotenv

from twilio.rest import Client
# from flask_sockets import Sockets
# import eventlet
# from eventlet import wsgi
# from eventlet import wrap_ssl
# import ssl


async_mode = 'None'
app = Flask(__name__)
socketio = SocketIO(app,async_mode=None)
# sockets = Sockets(app)
# socketio = SocketIO(app, cors_allowed_origins='*')
load_dotenv()
CORS(app)
UPLOAD_FOLDER = './uploads/'
TEMP_FOLDER = './temp/'
SQL_USERNAME = os.getenv('SQL_USERNAME')
SQL_PASSWORD = os.getenv('SQL_PASSWORD')
SQL_DB = os.getenv('SQL_DB')
twilio_account_sid = os.getenv('TWILIO_SID')
twilio_auth_token = os.getenv('TWILIO_TOKEN')
twilio_services = os.getenv('TWILIO_SERVICES')
client = Client(twilio_account_sid, twilio_auth_token)
ALLOWED_EXTENSIONS = {'mp4', 'avi'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
blueprint = Blueprint('api', __name__, url_prefix='/api')
app.register_blueprint(blueprint)

class NoneAlgorithm(jwt.algorithms.Algorithm):
    def prepare_key(self, key):
        pass

    def sign(self, msg, key):
        return b''

    def verify(self, msg, key, sig):
        return sig == b''

jwt.unregister_algorithm('none')
jwt.register_algorithm('none', NoneAlgorithm())

authorizations = {
    "Bearer": {
        "type": "apiKey", 
        "name": "Authorization", 
        "in": "header",
        "description": "Gunakan prefix <b><i>Bearer</b></i>, Contoh <b>Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIx</b>"
        }
    }

api = Api(
    app,
    authorizations=authorizations,
    title='ApiDocs',
    version='1.0',
    description='Preg-Fit API Documentation',
    prefix='/api'
    )



app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql://{SQL_USERNAME}:{SQL_PASSWORD}@127.0.0.1:3306/{SQL_DB}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ECHO"] = True

db = SQLAlchemy(app)

def point():
    x = [
        'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22',
        'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33',
    ]
    y = [
        'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20', 'y21', 'y22',
        'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29', 'y30', 'y31', 'y32', 'y33',
    ]
    z = [
        'z12', 'z13', 'z14', 'z15', 'z16', 'z17', 'z18', 'z19', 'z20', 'z21', 'z22',
        'z23', 'z24', 'z25', 'z26', 'z27', 'z28', 'z29', 'z30', 'z31', 'z32', 'z33',
    ]
    v = [
        'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22',
        'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33',
    ]
    coords = [x, y, z, v]
    return coords

def base64_to_image(base64_string):
    # Extract the base64 encoded binary data from the input string
    base64_data = base64_string.split(",")[1]
    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)
    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def is_video_file(file_storage):
    video_formats = ['mp4', 'avi', 'mov', 'mkv']  # Add more video formats as needed
    
    file_type = imghdr.what(file_storage.stream)
    if file_type is not None and file_type in video_formats:
        return True
    else:
        return False
    
def clasify_video(cap, upload):
    if (cap.isOpened() == False):
                print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_fps = input_fps - 1
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'video_w: {w}, video_h: {h}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if upload:
        out = cv2.VideoWriter(upload, fourcc, output_fps, (340, 180))

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()


            if ret == True:
                count += 1
                if count < input_fps/3:
                    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(frame, (340, 180), interpolation=cv2.INTER_LINEAR)
                    out.write(image)
                    continue
                else:
                    count = 0
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (340, 180), interpolation=cv2.INTER_LINEAR)

                    image.flags.writeable = False
                    results = holistic.process(image)
                    # Recolor image back to BGR for rendering
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # Pose Detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                            )
                    # Export coordinates
                    try:
                        # Extract Pose landmarks
                        pose = results.pose_landmarks.landmark
                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                        # Concate rows
                        row = pose_row
                        # print(f'type: {type(row)}, \n{row}')
                        # point[11] - point[32] input to tflite model.
                        coords = point()
                        specify_float = 8
                        dict_p12_to_p33 = {
                            # x12 to x33
                            coords[0][0]:round(row[44], specify_float),
                            coords[0][1]:round(row[48], specify_float),
                            coords[0][2]:round(row[52], specify_float),
                            coords[0][3]:round(row[56], specify_float),
                            coords[0][4]:round(row[60], specify_float),
                            coords[0][5]:round(row[64], specify_float),
                            coords[0][6]:round(row[68], specify_float),
                            coords[0][7]:round(row[72], specify_float),
                            coords[0][8]:round(row[76], specify_float),
                            coords[0][9]:round(row[80], specify_float),
                            coords[0][10]:round(row[84], specify_float),
                            coords[0][11]:round(row[88], specify_float),
                            coords[0][12]:round(row[92], specify_float),
                            coords[0][13]:round(row[96], specify_float),
                            coords[0][14]:round(row[100], specify_float),
                            coords[0][15]:round(row[104], specify_float),
                            coords[0][16]:round(row[108], specify_float),
                            coords[0][17]:round(row[112], specify_float),
                            coords[0][18]:round(row[116], specify_float),
                            coords[0][19]:round(row[120], specify_float),
                            coords[0][20]:round(row[124], specify_float),
                            coords[0][21]:round(row[128], specify_float),

                            # y12 to y33
                            coords[1][0]:round(row[45], specify_float),
                            coords[1][1]:round(row[49], specify_float),
                            coords[1][2]:round(row[53], specify_float),
                            coords[1][3]:round(row[57], specify_float),
                            coords[1][4]:round(row[61], specify_float),
                            coords[1][5]:round(row[65], specify_float),
                            coords[1][6]:round(row[69], specify_float),
                            coords[1][7]:round(row[73], specify_float),
                            coords[1][8]:round(row[77], specify_float),
                            coords[1][9]:round(row[81], specify_float),
                            coords[1][10]:round(row[85], specify_float),
                            coords[1][11]:round(row[89], specify_float),
                            coords[1][12]:round(row[93], specify_float),
                            coords[1][13]:round(row[97], specify_float),
                            coords[1][14]:round(row[101], specify_float),
                            coords[1][15]:round(row[105], specify_float),
                            coords[1][16]:round(row[109], specify_float),
                            coords[1][17]:round(row[113], specify_float),
                            coords[1][18]:round(row[117], specify_float),
                            coords[1][19]:round(row[121], specify_float),
                            coords[1][20]:round(row[125], specify_float),
                            coords[1][21]:round(row[129], specify_float),

                            # z12 to z33
                            coords[2][0]:round(row[46], specify_float),
                            coords[2][1]:round(row[50], specify_float),
                            coords[2][2]:round(row[54], specify_float),
                            coords[2][3]:round(row[58], specify_float),
                            coords[2][4]:round(row[62], specify_float),
                            coords[2][5]:round(row[66], specify_float),
                            coords[2][6]:round(row[70], specify_float),
                            coords[2][7]:round(row[74], specify_float),
                            coords[2][8]:round(row[78], specify_float),
                            coords[2][9]:round(row[82], specify_float),
                            coords[2][10]:round(row[86], specify_float),
                            coords[2][11]:round(row[90], specify_float),
                            coords[2][12]:round(row[94], specify_float),
                            coords[2][13]:round(row[98], specify_float),
                            coords[2][14]:round(row[102], specify_float),
                            coords[2][15]:round(row[106], specify_float),
                            coords[2][16]:round(row[110], specify_float),
                            coords[2][17]:round(row[114], specify_float),
                            coords[2][18]:round(row[118], specify_float),
                            coords[2][19]:round(row[122], specify_float),
                            coords[2][20]:round(row[126], specify_float),
                            coords[2][21]:round(row[130], specify_float),

                            # v12 to v33
                            coords[3][0]:round(row[47], specify_float),
                            coords[3][1]:round(row[51], specify_float),
                            coords[3][2]:round(row[55], specify_float),
                            coords[3][3]:round(row[59], specify_float),
                            coords[3][4]:round(row[63], specify_float),
                            coords[3][5]:round(row[67], specify_float),
                            coords[3][6]:round(row[71], specify_float),
                            coords[3][7]:round(row[75], specify_float),
                            coords[3][8]:round(row[79], specify_float),
                            coords[3][9]:round(row[83], specify_float),
                            coords[3][10]:round(row[87], specify_float),
                            coords[3][11]:round(row[91], specify_float),
                            coords[3][12]:round(row[95], specify_float),
                            coords[3][13]:round(row[99], specify_float),
                            coords[3][14]:round(row[103], specify_float),
                            coords[3][15]:round(row[107], specify_float),
                            coords[3][16]:round(row[111], specify_float),
                            coords[3][17]:round(row[115], specify_float),
                            coords[3][18]:round(row[119], specify_float),
                            coords[3][19]:round(row[123], specify_float),
                            coords[3][20]:round(row[127], specify_float),
                            coords[3][21]:round(row[131], specify_float),
                        }
                        input_dict = {name: np.expand_dims(np.array(value, dtype=np.float32), axis=0) for name, value in dict_p12_to_p33.items()}
                        # Make Detections.
                        model = tf.keras.models.load_model('./model/model_new.h5')
                        result = model.predict(input_dict)
                        result = result[0]
                        body_language_class = np.argmax(result)
                        # body_language_prob = round(result[np.argmax(result)], 2)*100
                        body_language_prob = result[np.argmax(result)]

                        # 1: cat-cow-pose, 2: child-pose, 4: lateral-leg-raise, 5: side-bend, 6: sideclamp , 7: savasana

                        if str(body_language_class) == '1':
                            pose_class = 'Cat Cow Pose'
                        elif str(body_language_class) == '2':
                            pose_class = 'Child Pose'
                        elif str(body_language_class) == '4':
                            pose_class = 'Lateral Leg Raise Pose'
                        elif str(body_language_class) == '5':
                            pose_class = 'Side Bend Pose'
                        elif str(body_language_class) == '6':
                            pose_class = 'Side Clamp Pose'
                        elif str(body_language_class) == '7':
                            pose_class = 'Savasana Pose'
                        else:
                            pose_class = 'GAK KEDETEKSI'

                        print(f'class: {body_language_class}, prob: {body_language_prob}')

                        # Show pose category near the ear.
                        coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                            [1280,480]
                        ).astype(int))
                        cv2.rectangle(image,
                                    (coords[0], coords[1]+5),
                                    (coords[0]+200, coords[1]-30),
                                    (245, 117, 16), -1)
                        cv2.putText(image, pose_class, coords,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        # Get status box
                        cv2.rectangle(image, (10,0), (310, 55), (0, 0, 0), -1)

                        # Display Class
                        cv2.putText(
                            image,
                            'CLASS: ', (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 0), 1, cv2.LINE_AA
                        )
                        cv2.putText(
                            image,
                            pose_class, (120, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA
                        )

                        # Display Probability
                        cv2.putText(
                            image,
                            'PROB: ', (15, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 0), 1, cv2.LINE_AA
                        )
                        cv2.putText(
                            image,
                            str(body_language_prob), (120, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA
                        )
                    except:
                        pass

                    out.write(image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            else:
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def decode(jwtToken):
    #Vulnerable
    # payload = jwt.decode(
    #     jwtToken,
    #     None,
    #     audience = [AUDIENCE_MOBILE],
    #     issuer = ISSUER,
    #     algorithms = ['none'],
    #     options = {"require": ["aud", "iss", "iat", "exp"]}
    #     )
    
    #Patch
    payload = jwt.decode(
        jwtToken,
        SECRET_KEY,
        audience = [AUDIENCE_MOBILE],
        issuer = ISSUER,
        algorithms = ['HS256'],
        options = {"require": ["aud", "iss", "iat", "exp"]}
        )
    
    return payload

def is_token_valid(token):
    try:
        payload = decode(token)
        print(payload)
        
        if 'exp' in payload:
            expiration = datetime.fromtimestamp(payload['exp'])
            if datetime.utcnow() > expiration:
                return False
        else:
            return False
        
        return True
    
    except jwt.ExpiredSignatureError:
        return False
    except jwt.InvalidTokenError:
        return False

def format_phone_number(no_hp):
    # Remove non-digit characters except for '+'
    no_hp = re.sub(r'[^\d+]', '', no_hp)
    
    # If the number already has a '+' prefix, return it as is
    if no_hp.startswith('+'):
        return no_hp
    
    # Handle different formats
    if no_hp.startswith('62'):
        return "+62" + no_hp[2:]
    elif no_hp.startswith('0'):
        return "+62" + no_hp[1:]
    elif no_hp.startswith('8'):
        return "+62" + no_hp
    else:
        return "+62" + no_hp

class User(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    no_hp = db.Column(db.String(13), unique=True, nullable=True)
    email = db.Column(db.String(32), nullable=True)
    nama = db.Column(db.String(64), nullable=True)
    usia_kandungan = db.Column(db.String(15), nullable=True)
    tanggal_lahir = db.Column(db.Date(), nullable=True)

class History(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), nullable=False)
    tanggal = db.Column(db.Date(), nullable=False)
    waktu = db.Column(db.String(50), nullable=False)
    jenis_yoga = db.Column(db.String(100), nullable=False)

class Feedback(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), nullable=False)
    komentar = db.Column(db.String(255), nullable=False)


SECRET_KEY = "117732f96ab4693ccdfffafb291d46d255fb519a0660a7f8d5bef8c68e6808f4"
ISSUER = "myFlaskWebService"
AUDIENCE_MOBILE = "myMobileApp"


parser4OTPsend = reqparse.RequestParser()
parser4OTPsend.add_argument('no_hp', type=str, location='json', required=True, help='Nomor HP')

parser4OTPverif = reqparse.RequestParser()
parser4OTPverif.add_argument('no_hp', type=str, location='json', required=True, help='Nomor HP')
parser4OTPverif.add_argument('otp', type=str, location='json', required=True, help='OTP')

parser4ChatBot = reqparse.RequestParser()
parser4ChatBot.add_argument('message', type=str, location='json', required=True, help='Message')

parser4SignUp = reqparse.RequestParser()
parser4SignUp.add_argument('no_hp', type=str, location='json', required=True, help='Nomor HP')

parser4CheckEmail = reqparse.RequestParser()
parser4CheckEmail.add_argument('email', type=str, location='json', required=True, help='Email')

parser4SignIn = reqparse.RequestParser()
parser4SignIn.add_argument('Authorization', type=str, location='headers', required=True, help='Nomor HP base64')


parser4UpdateUser = reqparse.RequestParser()
parser4UpdateUser.add_argument('no_hp', type=str, location='json', required=False, help='Nomor HP')
parser4UpdateUser.add_argument('email', type=str, location='json', required=False, help='Email')
parser4UpdateUser.add_argument('nama', type=str, location='json', required=False, help='Nama Lengkap')
parser4UpdateUser.add_argument('usia_kandungan', type=str, location='json', required=False, help='Usia Kandungan')
parser4UpdateUser.add_argument('tanggal_lahir', type=str, location='json', required=False, help='Tanggal Lahir')

parser4History = reqparse.RequestParser()
parser4History.add_argument('tanggal', type=str, location='json', required=True, help='Tanggal')
parser4History.add_argument('waktu', type=str, location='json', required=True, help='Waktu')
parser4History.add_argument('jenis_yoga', type=str, location='json', required=True, help='Jenis Yoga')

parser4Feedback = reqparse.RequestParser()
parser4Feedback.add_argument('komentar', type=str, location='json', required=True, help='Komentar')

parser4UpdateNohp = reqparse.RequestParser()
parser4UpdateNohp.add_argument('no_hp_baru', type=str, location='json', required=False, help='Nomor HP')
parser4UpdateNohp.add_argument('email', type=str, location='json', required=False, help='Email')

upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)

@api.route('/upload')
class Upload(Resource):
    @api.doc(security='Bearer')
    @api.expect(upload_parser)
    def post(self):
        start_time_1 = time.time()
        args = upload_parser.parse_args()
        uploaded_file = args['file']
        auth = request.headers.get('Authorization')
        try:
            jwtToken = auth[7:]
            payload = decode(jwtToken)
            
            user = db.session.execute(db.select(User).filter_by(id=payload['user_id'])).first()
            user = user[0]

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = user.nama + '_' + str(user.id) + timestamp + '.mp4'
            result_out = UPLOAD_FOLDER + filename
            temp_file = TEMP_FOLDER + filename
            if not uploaded_file.content_type.startswith('video/'):
                return {'message': 'Tipe file tidak valid, pastikan upload file video mom'}, 400
            uploaded_file.save(temp_file)
            cap = cv2.VideoCapture(temp_file)
            end_time = time.time()

            start_time = time.time()
            clasify_video(cap, result_out)
            end_time = time.time()
            execution_time = end_time - start_time
            print("Execution time:", execution_time, "seconds")
            
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return {'url': filename}, 201

        except:
            return {'message': 'Token tidak valid, silahkan masuk dulu mom'}, 401

@api.route('/update_nohp')
class Update_Nohp(Resource):
    @api.expect(parser4UpdateNohp)
    def post(self):
        args = parser4UpdateNohp.parse_args()
        email = args['email']
        no_hp = args['no_hp_baru']

        
        user = db.session.execute(db.select(User).filter_by(email=email)).first()
        user = user[0]

        if not no_hp:
            return {
                'message': 'No HP tidak boleh kosong'
            }, 400

        try:
            user.no_hp = no_hp

            db.session.commit()

            return {
                'message' : 'No HP berhasil diubah'
            }
        
        except:
            return {
                'message': 'No HP tidak dapat digunakan, atau sudah digunakan pengguna lain'
            }, 500


@api.route('/check_token')
class C_Token_Route(Resource):
    @api.doc(security='Bearer')
    @api.response(200, 'OK')
    def get(self):
        auth = request.headers.get('Authorization')

        jwtToken = auth[7:]

        is_valid = is_token_valid(jwtToken)
        if is_valid:
            return {
                'message' : 'Token valid!'
            }, 200
        else:
            return {'message': 'Token tidak valid, silahkan masuk dulu mom'}, 401


@api.route('/feedback')
class Feedback_Route(Resource):
    @api.doc(security='Bearer')
    @api.expect(parser4Feedback)
    @api.response(201, 'Created')
    def post(self):
        auth = request.headers.get('Authorization')
        args = parser4Feedback.parse_args()
        komentar = args['komentar']

        jwtToken = auth[7:]

        try:
            payload = decode(jwtToken)
            
            user = db.session.execute(db.select(User).filter_by(id=payload['user_id'])).first()
            user = user[0]

            if not komentar:
                return {
                    'message': 'Komentar tidak boleh kosong'
                }, 400

            feedback = Feedback()
            feedback.user_id = user.id
            feedback.komentar = komentar

            db.session.add(feedback)
            db.session.commit()

            return {
                'message' : 'Feedback berhasil ditambahkan'
            }, 201

        except:
            return {'message': 'Token tidak valid, silahkan masuk dulu mom'}, 401
        

@api.route('/history')
class History_Route(Resource):
    @api.doc(security='Bearer')
    @api.expect(parser4History)
    @api.response(201, 'Created')
    def post(self):
        auth = request.headers.get('Authorization')
        args = parser4History.parse_args()
        waktu = args['waktu']
        jenis_yoga = args['jenis_yoga']
        tanggal = args['tanggal']
        tanggal = tanggal.split("/")
        tanggal = date(int(tanggal[2]),int(tanggal[1]),int(tanggal[0]))

            # {
            # "tanggal": "01/01/2023",
            # "waktu": "5 Menit",
            # "jenis_yoga": "Bird Dog Pose"
            # }

        jwtToken = auth[7:]

        try:
            payload = decode(jwtToken)
            
            user = db.session.execute(db.select(User).filter_by(id=payload['user_id'])).first()
            user = user[0]

            if not waktu:
                return {
                    'message': 'Waktu tidak boleh kosong'
                }, 400
            
            if not jenis_yoga:
                return {
                    'message': 'Jenis yoga tidak boleh kosong'
                }, 400
            
            if not tanggal:
                return {
                    'message': 'Tanggal tidak boleh kosong'
                }, 400

            history = History()
            history.user_id = user.id
            history.tanggal = tanggal
            history.waktu = waktu
            history.jenis_yoga = jenis_yoga

            db.session.add(history)
            db.session.commit()

            return {
                'message' : 'History berhasil ditambahkan'
            }, 201

        except:
            return {'message': 'Token tidak valid, silahkan masuk dulu mom'}, 401
    
    @api.doc(security='Bearer')
    def get(self):
        auth = request.headers.get('Authorization')

        jwtToken = auth[7:]

        try:
            payload = decode(jwtToken)
            
            histories = db.session.execute(db.select(History).filter_by(user_id=payload['user_id'])).all()
            result = []
            for history in histories:
                history = history._asdict()['History']
                history.tanggal = dumps(history.tanggal, default=str).replace('"','')
                hist = {
                    'id': history.id,
                    'user_id': history.user_id,
                    'tanggal': history.tanggal,
                    'waktu': history.waktu,
                    'jenis_yoga': history.jenis_yoga
                }
                result.append(hist)


        except:
            return {'message': 'Token tidak valid, silahkan masuk dulu mom'}, 401

        return result[::-1], 200
    

@api.route('/popular')
class Popular_Route(Resource):
    @api.doc(security='Bearer')
    def get(self):
        auth = request.headers.get('Authorization')

        jwtToken = auth[7:]

        try:
            payload = decode(jwtToken)
            payload = decode(jwtToken)
            
            user = db.session.execute(db.select(User).filter_by(id=payload['user_id'])).first()
            user = user[0]

            if user:
                count_query = db.session.query(History.jenis_yoga, func.count(History.jenis_yoga).label('count')).group_by(History.jenis_yoga).order_by(func.count(History.jenis_yoga).desc())

                top_3_popular_jenis_yoga = count_query.limit(3).all()

                total_count = db.session.query(func.count(History.jenis_yoga)).scalar()

                result = []
                for jenis_yoga, count in top_3_popular_jenis_yoga:
                    popularity_percentage = int((count / total_count) * 100)
                    result.append({"jenis_yoga": jenis_yoga, "popularity_percentage": popularity_percentage})

        except:
            return {'message': 'Token tidak valid, silahkan masuk dulu mom'}, 401

        return result, 200


@api.route('/chatbot')
class Chatbot_Route(Resource):
    @api.expect(parser4ChatBot)
    @api.response(200, 'OK')
    def post(self):
        args = parser4ChatBot.parse_args()
        message = args['message']
        
        input_text_tokenized = bert_tokenizer.encode(message,
                                              truncation=True,
                                              padding='max_length',
                                               max_length = 20,
                                              return_tensors='tf')

        bert_predict = bert_load_model(input_text_tokenized) 
        bert_output = tf.nn.softmax(bert_predict[0], axis=-1)         # Softmax function untuk mendapatkan hasil klasifikasi
        output = tf.argmax(bert_output, axis=1)

        response_tag = labelencoder.inverse_transform(output.numpy().flatten())[0]
        response = random.choice(responses[response_tag])

        if response:
            return response, 200

        return {
            'message' : 'Maaf BellyBot tidak dapat melakukan verifikasi saat ini, silahkan coba beberapa saat lagi. Terima kasih',
            'value': 99
        }, 200  


@api.route('/check_no')
class Check_No_Route(Resource):
    @api.expect(parser4SignUp)
    @api.response(200, 'OK')
    def post(self):
        args = parser4SignUp.parse_args()
        no_hp = args['no_hp']

        if no_hp.startswith("0"):
            no_hp = no_hp[1:]
        
        result = db.session.execute(db.select(User).filter((User.no_hp == f'62{no_hp}') | (User.no_hp == no_hp)))
        user = result.fetchone()
        if user:
            return {
                'message': 'Nomor HP sudah digunakan, silahkan langsung masuk aja mom!'
            }, 409

        return {
            'message' : 'Nomor hp belum terdaftar'
        }, 200  

@api.route('/send_otp')
class Send_OTP_Route(Resource):
    @api.expect(parser4OTPsend)
    @api.response(200, 'OK')
    def post(self):
        args = parser4OTPsend.parse_args()
        no_hp = format_phone_number(args['no_hp'])

        verification = client.verify.v2.services(twilio_services).verifications.create(to=no_hp, channel='sms')
        
        if verification.status == "pending":
            return {
                'message': 'OTP berhasil dikirim mom!'
            }, 200

        return {
            'message' : 'Gagal kirim OTP mom!'
        }, 500  

@api.route('/verif_otp')
class Verif_OTP_Route(Resource):
    @api.expect(parser4OTPverif)
    @api.response(200, 'OK')
    def post(self):
        args = parser4OTPverif.parse_args()
        no_hp = format_phone_number(args['no_hp'])
        otp = args['otp']

        verification_check = client.verify.v2.services(twilio_services).verification_checks.create(to=no_hp, code=otp)
        return verification_check.status
        # if verification_check.status.valid:
        #     return {
        #         'message': 'OTP valid mom!'
        #     }, 200

        # return {
        #     'message' : 'OTP tidak valid mom!'
        # }, 400  

@api.route('/check_email')
class C_Email_Route(Resource):
    @api.expect(parser4CheckEmail)
    @api.response(200, 'OK')
    def post(self):
        args = parser4CheckEmail.parse_args()
        email = args['email']
        
        user = db.session.execute(db.select(User).filter_by(email=email)).first()
        if user:
            return {
                'message': 'Email terdaftar!'
            }, 409

        return {
            'message' : 'Email tidak terdaftar'
        }, 200  

@api.route('/users',methods=['GET', 'POST', 'PUT'])
class User_Route(Resource):
    @api.expect(parser4SignUp)
    @api.response(201, 'Created')
    def post(self):
        args = parser4SignUp.parse_args()
        no_hp = args['no_hp']
        if no_hp.startswith("0"):
            no_hp = f'62{no_hp[1:]}'
        
        user = db.session.execute(db.select(User).filter_by(no_hp=no_hp)).first()
        if user:
            return {
                'message': 'Nomor HP sudah digunakan, silahkan langsung masuk aja mom!'
            }, 409
        
        user = User()
        user.no_hp = no_hp
        user.tanggal_lahir = '1999-01-01'

        db.session.add(user)
        db.session.commit()

        return {
            'message' : 'Berhasil daftar mom'
        }, 201
    
    @api.doc(security='Bearer')
    def get(self):
        auth = request.headers.get('Authorization')

        jwtToken = auth[7:]

        try:
            payload = decode(jwtToken)
            
            user = db.session.execute(db.select(User).filter_by(id=payload['user_id'])).first()
            user = user[0]
            user.tanggal_lahir = dumps(user.tanggal_lahir, default=str).replace('"','')

        except:
            return {'message': 'Token tidak valid, silahkan masuk dulu mom'}, 401

        return {
            'id': user.id,
            'no_hp': user.no_hp,
            'email': user.email,
            'nama': user.nama,
            'usia_kandungan': user.usia_kandungan,
            'tanggal_lahir': user.tanggal_lahir
        }, 200

    @api.doc(security='Bearer')
    @api.expect(parser4UpdateUser)
    def put(self):
        print(request.json)
        try:
            args = request.get_json()
        except:
            return 1
        auth = request.headers.get('Authorization')
        jwtToken = auth[7:]
        no_hp = args['no_hp']
        email = args['email']
        nama = args['nama']
        if args['usia_kandungan']:
            usia_kandungan = args['usia_kandungan']
        
        if args['tanggal_lahir']:
            tanggal_lahir = args['tanggal_lahir']
            tanggal_lahir = tanggal_lahir.split("/")
            tanggal_lahir = date(int(tanggal_lahir[2]),int(tanggal_lahir[1]),int(tanggal_lahir[0]))

        try:
            payload = decode(jwtToken)
            userdb = db.session.execute(db.select(User).filter_by(id=payload['user_id'])).first()
            userdb = userdb[0]

        except:
            return {'message': 'Token tidak valid, silahkan masuk dulu mom'}, 401

        isUpdate = False
        if no_hp:
            userdb.no_hp = no_hp
            isUpdate = True
        if email:
            userdb.email = email
            isUpdate = True
        if nama:
            userdb.nama = nama
            isUpdate = True
        if args['usia_kandungan']:
            userdb.usia_kandungan = usia_kandungan
            isUpdate = True
        if args['tanggal_lahir']:
            userdb.tanggal_lahir = tanggal_lahir
            isUpdate = True

        if isUpdate:
            db.session.commit()

        return args, 200

@api.route('/signin')
class SignIn(Resource):
    @api.expect(parser4SignIn)
    def post(self):
        args = parser4SignIn.parse_args()
        basicAuth = args['Authorization']

        base64Msg = basicAuth[6:]
        msgBytes = base64Msg.encode('ascii')
        base64Bytes = base64.b64decode(msgBytes)
        no_hp = base64Bytes.decode('ascii')
        if no_hp.startswith("0"):
            no_hp = no_hp[1:]

        if not base64Msg:
            return {
                'message': 'Silahkan masukkan no HPnya dulu mom'
            }, 400
        
        result = db.session.execute(db.select(User).filter((User.no_hp == f'62{no_hp}') | (User.no_hp == no_hp)))
        user = result.fetchone()
        if not user:
            return {
                'message': 'No HP mom belum terdaftar di Preg-Fit, mom bisa daftar dulu'
            }, 400

        else:
            user = user[0]
            payload = {
                'user_id': user.id,
                'no_hp': user.no_hp,
                'aud': AUDIENCE_MOBILE,
                'iss': ISSUER,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=5)
            }

            #Vulnerable
            # token = jwt.encode(payload, None, algorithm='none')

            #PATCH
            token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')

            return {
                'token': token
            }, 200
        
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit("my response", {"data": "Connected"})   

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on_error()
def handle_error(e):
    print('An error occurred:', str(e))

@socketio.on('message')
def handle_message(data):
    print('Received message:', data)
    emit('response', 'Server received message: ' + data)

@socketio.on("image")
def handle_image(imageData):
    try:
        image_data_bytes = base64.b64decode(imageData['imageData'])
        image_array = np.frombuffer(image_data_bytes, dtype=np.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
        mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (340, 180), interpolation=cv2.INTER_LINEAR)

            # Make Detections
            # start_time = time.time()
            results = holistic.process(image)
            # end_time = time.time()

            # execution_time = end_time - start_time
            # print("Execution time:", execution_time, "seconds")

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            # Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            # Export coordinates
            try:
          # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Concate rows
                row = pose_row
                # print(f'type: {type(row)}, \n{row}')

                # point[11] - point[32] input to tflite model.
                coords = point()
                specify_float = 8

                dict_p12_to_p33 = {
                    # x12 to x33
                    coords[0][0]:round(row[44], specify_float),
                    coords[0][1]:round(row[48], specify_float),
                    coords[0][2]:round(row[52], specify_float),
                    coords[0][3]:round(row[56], specify_float),
                    coords[0][4]:round(row[60], specify_float),
                    coords[0][5]:round(row[64], specify_float),
                    coords[0][6]:round(row[68], specify_float),
                    coords[0][7]:round(row[72], specify_float),
                    coords[0][8]:round(row[76], specify_float),
                    coords[0][9]:round(row[80], specify_float),
                    coords[0][10]:round(row[84], specify_float),
                    coords[0][11]:round(row[88], specify_float),
                    coords[0][12]:round(row[92], specify_float),
                    coords[0][13]:round(row[96], specify_float),
                    coords[0][14]:round(row[100], specify_float),
                    coords[0][15]:round(row[104], specify_float),
                    coords[0][16]:round(row[108], specify_float),
                    coords[0][17]:round(row[112], specify_float),
                    coords[0][18]:round(row[116], specify_float),
                    coords[0][19]:round(row[120], specify_float),
                    coords[0][20]:round(row[124], specify_float),
                    coords[0][21]:round(row[128], specify_float),

                    # y12 to y33
                    coords[1][0]:round(row[45], specify_float),
                    coords[1][1]:round(row[49], specify_float),
                    coords[1][2]:round(row[53], specify_float),
                    coords[1][3]:round(row[57], specify_float),
                    coords[1][4]:round(row[61], specify_float),
                    coords[1][5]:round(row[65], specify_float),
                    coords[1][6]:round(row[69], specify_float),
                    coords[1][7]:round(row[73], specify_float),
                    coords[1][8]:round(row[77], specify_float),
                    coords[1][9]:round(row[81], specify_float),
                    coords[1][10]:round(row[85], specify_float),
                    coords[1][11]:round(row[89], specify_float),
                    coords[1][12]:round(row[93], specify_float),
                    coords[1][13]:round(row[97], specify_float),
                    coords[1][14]:round(row[101], specify_float),
                    coords[1][15]:round(row[105], specify_float),
                    coords[1][16]:round(row[109], specify_float),
                    coords[1][17]:round(row[113], specify_float),
                    coords[1][18]:round(row[117], specify_float),
                    coords[1][19]:round(row[121], specify_float),
                    coords[1][20]:round(row[125], specify_float),
                    coords[1][21]:round(row[129], specify_float),

                    # z12 to z33
                    coords[2][0]:round(row[46], specify_float),
                    coords[2][1]:round(row[50], specify_float),
                    coords[2][2]:round(row[54], specify_float),
                    coords[2][3]:round(row[58], specify_float),
                    coords[2][4]:round(row[62], specify_float),
                    coords[2][5]:round(row[66], specify_float),
                    coords[2][6]:round(row[70], specify_float),
                    coords[2][7]:round(row[74], specify_float),
                    coords[2][8]:round(row[78], specify_float),
                    coords[2][9]:round(row[82], specify_float),
                    coords[2][10]:round(row[86], specify_float),
                    coords[2][11]:round(row[90], specify_float),
                    coords[2][12]:round(row[94], specify_float),
                    coords[2][13]:round(row[98], specify_float),
                    coords[2][14]:round(row[102], specify_float),
                    coords[2][15]:round(row[106], specify_float),
                    coords[2][16]:round(row[110], specify_float),
                    coords[2][17]:round(row[114], specify_float),
                    coords[2][18]:round(row[118], specify_float),
                    coords[2][19]:round(row[122], specify_float),
                    coords[2][20]:round(row[126], specify_float),
                    coords[2][21]:round(row[130], specify_float),

                    # v12 to v33
                    coords[3][0]:round(row[47], specify_float),
                    coords[3][1]:round(row[51], specify_float),
                    coords[3][2]:round(row[55], specify_float),
                    coords[3][3]:round(row[59], specify_float),
                    coords[3][4]:round(row[63], specify_float),
                    coords[3][5]:round(row[67], specify_float),
                    coords[3][6]:round(row[71], specify_float),
                    coords[3][7]:round(row[75], specify_float),
                    coords[3][8]:round(row[79], specify_float),
                    coords[3][9]:round(row[83], specify_float),
                    coords[3][10]:round(row[87], specify_float),
                    coords[3][11]:round(row[91], specify_float),
                    coords[3][12]:round(row[95], specify_float),
                    coords[3][13]:round(row[99], specify_float),
                    coords[3][14]:round(row[103], specify_float),
                    coords[3][15]:round(row[107], specify_float),
                    coords[3][16]:round(row[111], specify_float),
                    coords[3][17]:round(row[115], specify_float),
                    coords[3][18]:round(row[119], specify_float),
                    coords[3][19]:round(row[123], specify_float),
                    coords[3][20]:round(row[127], specify_float),
                    coords[3][21]:round(row[131], specify_float),
                }
                #input coordinat to predict
                input_dict = {name: np.expand_dims(np.array(value, dtype=np.float32), axis=0) for name, value in dict_p12_to_p33.items()}
                # print(input_dict)

                # Make Detections
                # result = tflite_inference(input=input_dict, model=model)
                model = tf.keras.models.load_model('./model/model_new.h5')
                result = model.predict(input_dict)
                result = result[0]
                # print(result)
                body_language_class = np.argmax(result)
                # body_language_prob = round(result[np.argmax(result)], 2)*100
                body_language_prob = result[np.argmax(result)]
                # print('asd')

                # 1: cat-cow-pose, 2: child-pose, 4: lateral-leg-raise, 5: side-bend, 6: sideclamp , 7: savasana

                if str(body_language_class) == '1':
                    pose_class = 'Cat Cow Pose'
                elif str(body_language_class) == '2':
                    pose_class = 'Child Pose'
                elif str(body_language_class) == '4':
                    pose_class = 'Lateral Leg Raise Pose'
                elif str(body_language_class) == '5':
                    pose_class = 'Side Bend Pose'
                elif str(body_language_class) == '6':
                    pose_class = 'Side Clamp Pose'
                elif str(body_language_class) == '7':
                    pose_class = 'Savasana Pose'
                else:
                    pose_class = 'GAK KEDETEKSI'

                print(f'class: {body_language_class}, prob: {body_language_prob}')

                # pose_class = 'asd'
                # body_language_prob = '0.01212'

                # Show pose category near the ear.
                coords = tuple(np.multiply(
                    np.array(
                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                    [1280,480]
                ).astype(int))

                # # cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度).
                # # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類).
                # cv2.rectangle(image,
                #             (coords[0], coords[1]+5),
                #             (coords[0]+200, coords[1]-30),
                #             (245, 117, 16), -1)
                # cv2.putText(image, pose_class, coords,
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # # Get status box
                # cv2.rectangle(image, (10,0), (310, 55), (0, 0, 0), -1)

                # # Display Class
                # cv2.putText(
                #     image,
                #     'CLASS: ', (15, 25),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.9, (255, 255, 0), 1, cv2.LINE_AA
                # )
                # cv2.putText(
                #     image,
                #     pose_class, (120, 25),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1, (255, 255, 255), 2, cv2.LINE_AA
                # )

                # # Display Probability
                # cv2.putText(
                #     image,
                #     'PROB: ', (15, 50),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.9, (255, 255, 0), 1, cv2.LINE_AA
                # )
                # cv2.putText(
                #     image,
                #     str(body_language_prob), (120, 50),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1, (255, 255, 255), 2, cv2.LINE_AA
                # )

            except:
                pass


        # print('Received image data length:', len(imageData))
        # image_data_bytes = base64.b64decode(imageData['imageData'])
        # image_array = np.frombuffer(image_data_bytes, dtype=np.uint8)
        # decoded_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

        # print('Decoded image array shape:', decoded_image.shape)
        # height, width, channels = decoded_image.shape

        # print('Height:', height)
        # print('Width:', width)
        # print('Channels:', channels)

        # expected_shape = (height, width, channels)
        # if decoded_image.shape != expected_shape:
        #     print(f"Error: Invalid image shape. Expected: {expected_shape}, Actual: {decoded_image.shape}")
        #     return

        # image_rgb = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)

        processed_image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        processed_image_data = base64.b64encode(processed_image_bytes).decode('utf-8')
        prob = str(body_language_prob)

        emit('imageResponse', {"imageData": processed_image_data,"pose_class": pose_class, "prob": prob})

    except Exception as e:
        print('Error processing image:', e)


application = app.wsgi_app     

if __name__ == '__main__':
    #Pretrained Model
    PRE_TRAINED_MODEL = 'indobenchmark/indobert-base-p2'

    #Load tokenizer dari pretrained model
    bert_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)

    # Load hasil fine-tuning
    bert_load_model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=6)

    #Load Model
    bert_load_model.load_weights('model/chatbot-model.h5')
    # wsgi_server = eventlet.listen(('0.0.0.0', 5000))

    # # Load the SSL/TLS certificate and key files
    # cert_path = 'cert.pem'
    # key_path = 'key.pem'

    # # Create an SSL context and load the certificate and key
    # ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    # ssl_context.load_cert_chain(cert_path, key_path)

    # # Wrap the WSGI server with SSL/TLS support
    # wrapped_server = wrap_ssl(wsgi_server, certfile=cert_path, keyfile=key_path, ssl_version=ssl.PROTOCOL_TLS)

    # # Start the server
    # wsgi.server(wrapped_server, application)

    # # Run the API web service on port 5000
    # app.run(debug=True, host='0.0.0.0', port=5000, ssl_context='adhoc', use_reloader=False)

    # # Create the WebSocket server on port 8000
    # websocket_app = Flask(__name__)
    # socketio = SocketIO(websocket_app, async_mode='eventlet')

    # @websocket_app.route('/')
    # def index():
    #     # WebSocket endpoint logic here
    #     pass

    # # Run the WebSocket server on port 8000
    # eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 8000)), websocket_app)

    # cert_path = 'cert.pem'
    # key_path = 'key.pem'

    # ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    # ssl_context.load_cert_chain(cert_path, key_path)

    # wsgi.server(eventlet.listen(('0.0.0.0', 5000)), application, certfile=cert_path, keyfile=key_path, server_side=True, ssl_version=ssl.PROTOCOL_TLS, debug=True)

    # wrapped_socket = wrap_ssl(eventlet.listen(('0.0.0.0', 5000)), application, certfile=cert_path, keyfile=key_path, server_side=True, ssl_version=ssl.PROTOCOL_TLS, debug=True)
    # wsgi.server(wrapped_socket)

    socketio.run(app, debug=True, host='0.0.0.0')


    # app.run(debug=True, host='0.0.0.0', ssl_context='adhoc')