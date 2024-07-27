from flask import Flask, Blueprint, request, jsonify
from flask_restx import Resource, Api, Namespace, reqparse
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
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
from twilio.base.exceptions import TwilioRestException
import pytz


async_mode = 'None'
app = Flask(__name__)
socketio = SocketIO(app,async_mode=None)
load_dotenv()
CORS(app)

SQL_USERNAME = os.getenv('SQL_USERNAME')
SQL_PASSWORD = os.getenv('SQL_PASSWORD')
SQL_DB = os.getenv('SQL_DB')

twilio_account_sid = os.getenv('TWILIO_SID')
twilio_auth_token = os.getenv('TWILIO_TOKEN')
twilio_services = os.getenv('TWILIO_SERVICES')
client = Client(twilio_account_sid, twilio_auth_token)


SECRET_KEY = os.getenv('APP_SECRET_KEY')
API_KEY = os.getenv('API_KEY')
ISSUER = "myFlaskWebService"
AUDIENCE_MOBILE = "myMobileApp"
blueprint = Blueprint('api', __name__, url_prefix='/api')
app.register_blueprint(blueprint)


app.config['MAIL_SERVER']= os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = 2525
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

local_timezone = pytz.timezone('Asia/Jakarta')

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
    doc='/docs',
    authorizations=authorizations,
    title='ApiDocs',
    version='2.0',
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
    
def decode(jwtToken):
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

def verif_otp(no_hp, otp):
    try:
        verification_check = client.verify.v2.services(twilio_services).verification_checks.create(to=no_hp, code=otp)
        if verification_check.valid:
            return True, None
        return False, 'OTP tidak valid mom!'
    except TwilioRestException as e:
        if e.code == 20404:
            print("Error: The requested resource was not found. Please check the Service SID.")
            return False, 'OTP tidak valid mom!'
        elif e.code == 60202:
            print("Error: Max check attempts reached. Please try again later.")
            return False, 'Terlalu banyak salah silahkan coba beberapa saat lagi mom!'
        else:
            print(f"Twilio error: {e.code} - {e.msg}")
            return False, 'Terjadi kesalahan pada sistem. Silahkan coba lagi nanti.'

def generate_token(user):
    payload = {
        'user_id': user.id,
        'no_hp': user.no_hp,
        'aud': AUDIENCE_MOBILE,
        'iss': ISSUER,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token

def sendMail(email, otp):
    msg = Message('Verifkasi OTP Preg-Fit', 
                  sender=('Aplikasi Preg-Fit', 'preg-fit@yopmail.com'), 
                  recipients=[email])
    msg.body = "Kode verifikasi OTP kamu adalah " + otp
    msg.html = '''<div style="max-width: 600px; margin: 20px auto; background-color: #ffffff; border-radius: 8px; overflow: hidden;">
        <div style="background-color: #fff;; padding: 0 20px; text-align: center; color: #7F91AA;">
            <img style="display: block; margin: 0 auto; width: 80px;" src="https://i.ibb.co.com/pzFpNnG/ic-logo.png" title="logo" alt="logo" data-bit="iit">
            <h1 style="margin-top: 0;">Preg-Fit</h1>
        </div>
        <div style="height: 1px; background-color: #ecf0f1; margin: 20px 0;"></div>
        <p>Hi Mom!<br/><br/>Ini dia kode rahasia kita. Jangan berikan kode ini ke siapa pun untuk alasan keamanan ya! <br/>Gunakan kode ini hanya untuk verifikasi akun <strong>Preg-Fit</strong> mom.<br/> Semoga hari mom menyenangkan yaa &#129392; &#129392;.</p><br> 
      <center><span style="background: red; padding: 5px 15px 5px 15px; font-size: 20px; font-weight: bold; border-radius: 20px; color: white; background-color: #2c3e50;">{otp}</span></center>
            <div style="height: 1px; background-color: #ecf0f1; margin: 20px 0;"></div>
        <div style="text-align: center; padding: 20px; background-color: #ecf0f1;">
            <p style="font-size: 14px; color: #03265B; margin: 0;">&copy; <strong>Preg-Fit</strong></p><div class="yj6qo"></div><div class="adL">
        </div></div><div class="adL">
    </div></div>'''.format(otp=otp)
    mail.send(msg)
    return "OTP Sent to" + email

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

class OtpMail(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(32), nullable=True)
    otp = db.Column(db.String(128), nullable=True)
    otp_expired_at = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = db.Column(db.DateTime, nullable=False)

parser4OTPsend = reqparse.RequestParser()
parser4OTPsend.add_argument('no_hp', type=str, location='json', required=True, help='Nomor HP')

parser4OTPverif = reqparse.RequestParser()
parser4OTPverif.add_argument('no_hp', type=str, location='json', required=True, help='Nomor HP')
parser4OTPverif.add_argument('otp', type=str, location='json', required=True, help='OTP')
parser4OTPverif.add_argument('action', type=int, location='json', required=True, help='Aksi')
parser4OTPverif.add_argument('email', type=str, location='json', required=False, help='Email')

parser4ChatBot = reqparse.RequestParser()
parser4ChatBot.add_argument('message', type=str, location='json', required=True, help='Message')

parser4CheckEmail = reqparse.RequestParser()
parser4CheckEmail.add_argument('email', type=str, location='json', required=True, help='Email')

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

parser4CheckNo = reqparse.RequestParser()
parser4CheckNo.add_argument('no_hp', type=str, location='json', required=True, help='Nomor HP')

parser4SendOtpMail = reqparse.RequestParser()
parser4SendOtpMail.add_argument('api-key', type=str, location='headers', required=True, help='API Key')
parser4SendOtpMail.add_argument('email', type=str, location='json', required=True, help='Email')

parser4VerifyOtpMail = reqparse.RequestParser()
parser4VerifyOtpMail.add_argument('api-key', type=str, location='headers', required=True, help='Api Key')
parser4VerifyOtpMail.add_argument('otp', type=str, location='json', required=True, help='OTP')
parser4VerifyOtpMail.add_argument('email', type=str, location='json', required=True, help='Email')


@api.route('/send_otp_mail')
class SendOTPMail_Route(Resource):
    @api.expect(parser4SendOtpMail, validate=True)
    @api.response(200, 'OK')
    def post(self):
        args = parser4SendOtpMail.parse_args()
        apiKey = args['api-key']
        email = args['email']

        # Dapatkan waktu saat ini dengan informasi zona waktu
        now = datetime.now(local_timezone)

        # Checking for API key validity
        if API_KEY != apiKey:
            return {'message': 'API KEY Invalid!'}, 400

        try:
            # Check OTP and its expiration
            checkOtp = OtpMail.query.filter(OtpMail.email == email).first()
            if checkOtp:
                if checkOtp.otp_expired_at.tzinfo is None:
                    checkOtp.otp_expired_at = local_timezone.localize(checkOtp.otp_expired_at)

                if checkOtp.otp_expired_at > now:
                    return {'message': 'OTP sudah kami kirim, cek email yuk'}, 409
                else:
                    db.session.delete(checkOtp)
                    db.session.commit()
            
            # Fetch user email
            user = User.query.filter_by(email=email).first()
            if not user:
                return {'message': 'User tidak ditemukan'}, 400
            
            email = user.email

            # Generate random 4-digit OTP
            otp = random.randint(100000, 999999)
            otp_expired_at = now + timedelta(minutes=5)
            
            # Add OTP record
            OTPMail = OtpMail()
            OTPMail.email = email
            OTPMail.otp = generate_password_hash(str(otp))
            OTPMail.otp_expired_at = otp_expired_at
            OTPMail.updated_at = now

            # Add the OTPMail object to the session
            db.session.add(OTPMail)

            # Commit the session
            db.session.commit()

            # Send OTP to email
            sendMail(email, str(otp))

        except Exception as e:
            return {'message': str(e)}, 500

        return {
            'message': 'Berhasil Send OTP',
            'data': {
                'otp_expired_at': otp_expired_at.strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': OTPMail.updated_at.strftime('%Y-%m-%d %H:%M:%S')
            }
        }, 200


@api.route('/verify_otp_mail')
class VerifyOtpMail_Route(Resource):
    @api.expect(parser4VerifyOtpMail, validate=True)
    @api.response(200, 'OK')
    def post(Self):
        args = parser4VerifyOtpMail.parse_args()
        apiKey = args['api-key']
        otp = args['otp']
        email = args['email']
        
        now = datetime.now(local_timezone)

        #checking for api key is valid
        if API_KEY != apiKey:
            return {'message':'API KEY Invalid!'}, 400
        
        try:
            #start transaction
            with db.session.begin():
        
                #check code otp or otp_expired_at < now
                checkOtp = OtpMail.query.filter(OtpMail.email==email).first()

                if checkOtp is None:
                    return {'message': 'Silahkan resend OTP'}, 400

                if checkOtp.otp_expired_at.tzinfo is None:
                    checkOtp.otp_expired_at = local_timezone.localize(checkOtp.otp_expired_at)

                if not check_password_hash(checkOtp.otp, otp):
                    return {'message': 'OTP Invalid'}, 400

                if checkOtp.otp_expired_at < now:
                    return {'message': 'OTP Expired!'}, 410
                        
                #get user by email associated with the OTP
                user = db.session.execute(db.select(User).filter_by(email=checkOtp.email)).first()
                if not user:
                    return {'message': 'Akun tidak ditemukan!'}, 400
                
                # Remove the OTP record after successful verification
                db.session.delete(checkOtp)
                

        except Exception as e:
            #rollback here
            return {'message': str(e)}, 500

        return {
            'message' : 'Berhasil Verifikasi OTP'
        }, 200


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
    @api.expect(parser4CheckNo)
    @api.response(200, 'OK')
    def post(self):
        args = parser4CheckNo.parse_args()
        no_hp = format_phone_number(args['no_hp'])

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
        action = args['action']
        email = args.get('email')

        if action == 0:
            user = db.session.execute(db.select(User).filter(User.no_hp == no_hp)).scalar()
            if not user:
                return {'message': 'No HP mom belum terdaftar di Preg-Fit, mom bisa daftar dulu'}, 400
            
            valid, error_message = verif_otp(no_hp, otp)
            if valid:
                token = generate_token(user)
                return {'token': token}, 200
            return {'message': error_message}, 400

        elif action == 1:
            user = db.session.execute(db.select(User).filter_by(no_hp=no_hp)).scalar()
            if user:
                return {'message': 'Nomor HP sudah digunakan, silahkan langsung masuk aja mom!'}, 409

            valid, error_message = verif_otp(no_hp, otp)
            if valid:
                new_user = User(no_hp=no_hp, tanggal_lahir='1999-01-01')
                db.session.add(new_user)
                db.session.commit()

                new_user = db.session.execute(db.select(User).filter(User.no_hp == no_hp)).scalar()
                token = generate_token(new_user)
                return {'token': token, 'message': 'Berhasil daftar mom'}, 201

            return {'message': error_message}, 400
        
        elif action == 2:
            user = db.session.execute(db.select(User).filter(User.no_hp == no_hp)).scalar()
            if user:
                return {'message': 'Nomor HP sudah digunakan, silahkan coba nomor lain!'}, 409

            valid, error_message = verif_otp(no_hp, otp)
            if valid:
                current_user = db.session.execute(db.select(User).filter_by(email=email)).scalar()
                current_user.no_hp = no_hp
                db.session.commit()

                token = generate_token(current_user)
                return {'token': token, 'message': 'No HP berhasil diubah'}, 200

            return {'message': error_message}, 400


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


@api.route('/users',methods=['GET', 'PUT'])
class User_Route(Resource):
    @api.response(200, 'OK')
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

        if email:
            existing_user = db.session.execute(db.select(User).filter_by(email=email)).first()
            if existing_user and existing_user[0].id != userdb.id:
                return {'message': 'Email sudah digunakan oleh pengguna lain mom.'}, 400

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

            results = holistic.process(image)
            
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

                # Make Detections
                model = tf.keras.models.load_model('./model/model_new.h5')
                result = model.predict(input_dict)
                result = result[0]
                
                body_language_class = np.argmax(result)
                
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

            except:
                pass

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
    bert_load_model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=18)

    #Load Model
    bert_load_model.load_weights('model/model_chatbot.h5')

    socketio.run(app, debug=True, host='0.0.0.0')