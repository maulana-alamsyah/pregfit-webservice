from flask import Flask, Blueprint, request, jsonify
from flask_restx import Resource, Api, Namespace, reqparse
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.security import generate_password_hash, check_password_hash
from datetime  import date, datetime, timedelta
import jwt
from json import dumps
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_sockets import Sockets
import eventlet
from eventlet import wsgi
from eventlet import wrap_ssl
import ssl


async_mode = 'None'
app = Flask(__name__)
sockets = SocketIO(app)
# sockets = Sockets(app)
# socketio = SocketIO(app, cors_allowed_origins='*')
CORS(app)
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



app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://root:@127.0.0.1:3306/pregfit_webservice"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ECHO"] = True

db = SQLAlchemy(app)

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


SECRET_KEY = "WhatEverYouWant!"
ISSUER = "myFlaskWebService"
AUDIENCE_MOBILE = "myMobileApp"


parser4SignUp = reqparse.RequestParser()
parser4SignUp.add_argument('no_hp', type=str, location='json', required=True, help='Nomor HP')

parser4CheckEmail = reqparse.RequestParser()
parser4CheckEmail.add_argument('email', type=str, location='json', required=True, help='Email')

parser4SignIn = reqparse.RequestParser()
parser4SignIn.add_argument('no_hp', type=str, location='json', required=True, help='Nomor HP')

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


# @socketio.on('my_custom_event')
# def handle_custom_event(data):
#     # Process WebSocket message
#     # Ensure that this route does not interfere with the API endpoint
#     print(data)
#     return data

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

        return result, 200
    

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


@api.route('/check_no')
class C_No_Route(Resource):
    @api.expect(parser4SignUp)
    @api.response(200, 'OK')
    def post(self):
        args = parser4SignUp.parse_args()
        no_hp = args['no_hp']
        
        user = db.session.execute(db.select(User).filter_by(no_hp=no_hp)).first()
        if user:
            return {
                'message': 'Nomor HP sudah digunakan, silahkan langsung masuk aja mom!'
            }, 409

        return {
            'message' : 'Nomor hp belum terdaftar'
        }, 200  

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

@api.route('/users')
class User_Route(Resource):
    @api.expect(parser4SignUp)
    @api.response(201, 'Created')
    def post(self):
        args = parser4SignUp.parse_args()
        no_hp = args['no_hp']
        
        user = db.session.execute(db.select(User).filter_by(no_hp=no_hp)).first()
        if user:
            return {
                'message': 'Nomor HP sudah digunakan, silahkan langsung masuk aja mom!'
            }, 409
        
        user = User()
        user.no_hp = no_hp

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
        args = parser4UpdateUser.parse_args()
        auth = request.headers.get('Authorization')
        jwtToken = auth[7:]
        no_hp = args['no_hp']
        email = args['email']
        nama = args['nama']
        usia_kandungan = args['usia_kandungan']
        tanggal_lahir = args['tanggal_lahir']
        if tanggal_lahir:
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
        if usia_kandungan:
            userdb.usia_kandungan = usia_kandungan
            isUpdate = True
        if tanggal_lahir:
            userdb.tanggal_lahir = tanggal_lahir
            isUpdate = True

        if isUpdate:
            db.session.commit()

        return {
            'message' : 'Data berhasil diupdate mom'
        }, 200

@api.route('/signin')
class SignIn(Resource):
    @api.expect(parser4SignIn)
    def post(self):
        args = parser4SignIn.parse_args()
        no_hp = args['no_hp']

        if not no_hp:
            return {
                'message': 'Silahkan masukkan no HPnya dulu mom'
            }, 400
        
        user = db.session.execute(db.select(User).filter_by(no_hp=no_hp)).first()
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
        
# @sockets.route('/ws')
# def handle_websocket(ws):
#     return 1

application = app.wsgi_app     

if __name__ == '__main__':
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

    # sockets.run(app, debug=True, host='0.0.0.0')


    app.run(debug=True, host='0.0.0.0', ssl_context='adhoc')