import os
import traceback
import json
import random
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from django.core.cache import cache
from django.core.mail import send_mail
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.tokens import default_token_generator
from django.conf import settings
from django.contrib.auth.models import User
from rest_framework import generics, status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
import google.generativeai as genai
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes

from .models import MedicalReport
from .serializers import MedicalReportSerializer, UserSerializer

# --- OCR and AI Configuration ---
# NOTE: Removed hardcoded path to Tesseract, assumes it's globally available
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_GEMINI_MODEL = os.getenv('GOOGLE_GEMINI_MODEL')

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


# --- Helper Functions ---
def extract_text_from_file(file_obj):
    """Extract text from an uploaded image or PDF file."""
    text = ""

    if file_obj.name.lower().endswith('.pdf'):
        images = convert_from_bytes(file_obj.read())
        for image in images:
            text += pytesseract.image_to_string(image)
    else:
        image = Image.open(file_obj)
        text = pytesseract.image_to_string(image)

    return text


def get_real_ai_analysis(report_text):
    """Send extracted text to Gemini AI for structured medical analysis."""
    print(f"DEBUG: Attempting to use API Key ending in ...{GOOGLE_API_KEY[-4:] if GOOGLE_API_KEY else 'None'}")

    if not GOOGLE_API_KEY:
        return {"error": "AI service is not configured."}

    model = genai.GenerativeModel(GOOGLE_GEMINI_MODEL)

    prompt = f"""
    Analyze the following medical report text and provide a structured response in JSON format.
    IMPORTANT: Do not use Markdown formatting (like **bold** or *italics*). Output only valid JSON.

    Required JSON Keys:
    - "description": A detailed description of findings.
    - "medicine_recommendation": List of suggested OTC medicines or basic treatments.
    - "home_remedies": List of lifestyle/diet remedies.
    - "precautions": List of precautions or things to avoid.
    - "nearby_specialist": Type of specialist to consult.
    - "emergency_video": A relevant YouTube emergency video URL.
    - "status": One of "Normal", "Action Needed", or "High Risk".

    Medical Report Text:
    ---
    {report_text}
    ---
    """

    try:
        response = model.generate_content(prompt)
        json_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_response)
    except Exception as e:
        print(f"AI generation failed: {e}")
        return {"error": f"Failed to get AI analysis. Details: {str(e)}"}


# --- Django Views ---

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (AllowAny,)
    serializer_class = UserSerializer

    def create(self, request, *args, **kwargs):
        existing_inactive_user = User.objects.filter(
            email__iexact=request.data.get('email'), is_active=False
        ).first()

        if existing_inactive_user:
            user = existing_inactive_user
        else:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            user = serializer.save()

        user.is_active = False
        user.save()

        otp = str(random.randint(100000, 999999))
        cache.set(f"otp_{user.email}", otp, timeout=300)

        email_body = f"""
        Hello {user.username},

        Thank you for registering. Your One-Time Password (OTP) is: {otp}
        This code will expire in 5 minutes.
        """

        try:
            send_mail(
                'Verify your email address for Arogya AI',
                email_body,
                settings.EMAIL_HOST_USER,
                [user.email],
                fail_silently=False,
            )
            print(f"Successfully sent OTP to {user.email}")
        except Exception as e:
            print(f"Failed to send OTP email: {e}")

        return Response(
            {'email': user.email, 'message': 'Registration successful. OTP sent for verification.'},
            status=status.HTTP_201_CREATED,
        )


class VerifyOTPView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        email = request.data.get('email')
        otp_entered = request.data.get('otp')

        if not email or not otp_entered:
            return Response({'error': 'Email and OTP are required.'}, status=status.HTTP_400_BAD_REQUEST)

        stored_otp = cache.get(f"otp_{email}")

        if not stored_otp:
            return Response({'error': 'OTP expired or invalid.'}, status=status.HTTP_400_BAD_REQUEST)

        if otp_entered == stored_otp:
            user_to_activate = User.objects.filter(email__iexact=email).order_by('-date_joined').first()
            if user_to_activate:
                user_to_activate.is_active = True
                user_to_activate.save()
                cache.delete(f"otp_{email}")
                return Response({'message': 'Email verified successfully.'}, status=status.HTTP_200_OK)
            return Response({'error': 'User not found or already active.'}, status=status.HTTP_400_BAD_REQUEST)

        return Response({'error': 'Invalid OTP.'}, status=status.HTTP_400_BAD_REQUEST)


class MedicalReportUploadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        file_obj = request.data.get('report_file')
        if not file_obj:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            extracted_text = extract_text_from_file(file_obj)
            if not extracted_text.strip():
                return Response({"error": "Could not read text from file."}, status=status.HTTP_400_BAD_REQUEST)

            analysis_results = get_real_ai_analysis(extracted_text)
            if "error" in analysis_results:
                return Response(analysis_results, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            report = MedicalReport.objects.create(
                user=request.user,
                report_file=file_obj,
                extracted_text=extracted_text,
                ai_description=analysis_results.get("description", ""),
                ai_medicine_rec=analysis_results.get("medicine_recommendation", ""),
                ai_remedies=analysis_results.get("home_remedies", ""),
                ai_precautions=analysis_results.get("precautions", ""),
                emergency_video_url=analysis_results.get("emergency_video", ""),
                status=analysis_results.get("status", "Pending"),
            )

            analysis_results['extracted_text'] = extracted_text
            return Response(analysis_results, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": f"Unexpected error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MedicalReportListCreateView(generics.ListAPIView):
    serializer_class = MedicalReportSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return MedicalReport.objects.filter(user=self.request.user).order_by('-created_at')


class MedicalReportDeleteView(generics.DestroyAPIView):
    serializer_class = MedicalReportSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return MedicalReport.objects.filter(user=self.request.user)


def _fetch_json(url, params=None, data=None, headers=None, timeout=12):
    if params:
        url = f"{url}?{urlencode(params)}"
    request_headers = headers or {}
    req = Request(url, data=data, headers=request_headers)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _build_address(tags):
    parts = []
    if tags.get("addr:housenumber"):
        parts.append(tags["addr:housenumber"])
    if tags.get("addr:street"):
        parts.append(tags["addr:street"])
    street_line = " ".join(parts).strip()
    city = tags.get("addr:city")
    state = tags.get("addr:state")
    postcode = tags.get("addr:postcode")
    address_parts = [p for p in [street_line, city, state, postcode] if p]
    return ", ".join(address_parts)


class NearbyDoctorsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        city = request.query_params.get("city", "").strip()
        if not city:
            return Response(
                {"error": "City is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        nominatim_headers = {
            "User-Agent": "ArogyaAI/1.0 (contact: support@arogya.ai)",
            "Accept-Language": "en",
        }
        overpass_headers = {
            "User-Agent": "ArogyaAI/1.0 (contact: support@arogya.ai)",
        }

        try:
            geocode = _fetch_json(
                "https://nominatim.openstreetmap.org/search",
                params={"q": city, "format": "json", "limit": 1},
                headers=nominatim_headers,
                timeout=10,
            )
        except (URLError, HTTPError) as err:
            return Response(
                {"error": f"Geocoding failed: {str(err)}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        if not geocode:
            return Response({"city": city, "doctors": []}, status=status.HTTP_200_OK)

        lat = geocode[0].get("lat")
        lon = geocode[0].get("lon")
        if not lat or not lon:
            return Response({"city": city, "doctors": []}, status=status.HTTP_200_OK)

        radius_m = 4500
        limit = 10
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="doctors"](around:{radius_m},{lat},{lon});
          node["healthcare"="doctor"](around:{radius_m},{lat},{lon});
          way["amenity"="doctors"](around:{radius_m},{lat},{lon});
          way["healthcare"="doctor"](around:{radius_m},{lat},{lon});
          relation["amenity"="doctors"](around:{radius_m},{lat},{lon});
          relation["healthcare"="doctor"](around:{radius_m},{lat},{lon});
        );
        out center {limit};
        """
        overpass_data = urlencode({"data": overpass_query}).encode("utf-8")

        overpass_endpoints = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.nchc.org.tw/api/interpreter",
            "https://overpass.openstreetmap.fr/api/interpreter",
        ]
        overpass_result = None
        last_error = None
        for endpoint in overpass_endpoints:
            try:
                overpass_result = _fetch_json(
                    endpoint,
                    data=overpass_data,
                    headers=overpass_headers,
                    timeout=16,
                )
                if overpass_result and overpass_result.get("elements"):
                    break
            except (URLError, HTTPError) as err:
                last_error = err
                continue

        if overpass_result is None:
            return Response(
                {"error": f"Nearby doctor search failed: {str(last_error)}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        doctors = []
        seen = set()
        for element in overpass_result.get("elements", []):
            tags = element.get("tags", {})
            name = tags.get("name") or tags.get("operator") or "Doctor"
            address = _build_address(tags)
            el_lat = element.get("lat") or element.get("center", {}).get("lat")
            el_lon = element.get("lon") or element.get("center", {}).get("lon")
            key = f"{name}|{address}|{el_lat}|{el_lon}"
            if key in seen:
                continue
            seen.add(key)
            doctors.append(
                {
                    "name": name,
                    "address": address,
                    "lat": el_lat,
                    "lon": el_lon,
                }
            )
            if len(doctors) >= 5:
                break

        return Response(
            {"city": city, "doctors": doctors},
            status=status.HTTP_200_OK,
        )


class ChatView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        report_text = request.data.get('report_text')
        chat_history = request.data.get('history', [])

        if not chat_history:
            return Response({"error": "No user question provided."}, status=status.HTTP_400_BAD_REQUEST)

        user_question = chat_history[-1]['parts'][0]['text']

        if not report_text or not user_question:
            return Response({"error": "Report text and question required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            model = genai.GenerativeModel(GOOGLE_GEMINI_MODEL)
            initial_context = {
                "role": "user",
                "parts": [
                    f"You are a helpful medical assistant. Here is a medical report: '{report_text}'. "
                    "Answer follow-up questions based only on this report."
                ],
            }
            context_response = {
                "role": "model",
                "parts": [
                    "Understood. I have the context and will answer accordingly."
                ],
            }
            history_for_chat = [initial_context, context_response] + chat_history[:-1]

            chat = model.start_chat(history=history_for_chat)
            response = chat.send_message(user_question)

            return Response({"response": response.text}, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": f"AI chat error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GeneralChatView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        chat_history = request.data.get('history', [])
        if not chat_history:
            return Response({"error": "No chat history provided."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            model = genai.GenerativeModel(GOOGLE_GEMINI_MODEL)
            user_question = chat_history[-1]['parts'][0]['text']
            history_for_chat = chat_history[:-1]

            system_instruction = {
                "role": "user",
                "parts": [
                    {"text": "You are a helpful medical assistant named Arogya AI. "
                             "Provide general information, not personal advice."}
                ],
            }
            context_setter = {
                "role": "model",
                "parts": [{"text": "Understood. I'm ready for your medical questions."}],
            }

            full_context_history = [system_instruction, context_setter] + history_for_chat
            chat = model.start_chat(history=full_context_history)
            response = chat.send_message(user_question)

            return Response({"reply": response.text}, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": f"AI service failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PasswordResetRequestView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get('email')
        user = User.objects.filter(email__iexact=email).first()

        if user and user.is_active:
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)
            reset_link = f"http://localhost:5173/reset-password/{uid}/{token}/"

            email_body = f"""
            Hello,
            Click below to reset your password for Arogya AI:
            {reset_link}
            If you did not request this, please ignore it.
            """

            try:
                send_mail(
                    'Arogya AI Password Reset Request',
                    email_body,
                    settings.EMAIL_HOST_USER,
                    [user.email],
                    fail_silently=False,
                )
            except Exception:
                traceback.print_exc()

        return Response(
            {'message': 'If the account exists, a password reset link was sent.'},
            status=status.HTTP_200_OK,
        )


class PasswordResetConfirmView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, uidb64, token):
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None

        if user and default_token_generator.check_token(user, token):
            password = request.data.get('password')
            if not password:
                return Response({'error': 'Password required.'}, status=status.HTTP_400_BAD_REQUEST)

            user.set_password(password)
            user.save()
            return Response({'message': 'Password reset successful.'}, status=status.HTTP_200_OK)

        return Response({'error': 'Invalid reset link.'}, status=status.HTTP_400_BAD_REQUEST)
