# api/urls.py
from django.urls import path
from .views import (
    MedicalReportListCreateView,
    RegisterView,
    MedicalReportUploadView,
    ChatView,
    GeneralChatView,
    NearbyDoctorsView,
    PasswordResetRequestView,
    PasswordResetConfirmView,
    VerifyOTPView,
    MedicalReportDeleteView
)

urlpatterns = [
    # Report Management
    path('reports/', MedicalReportListCreateView.as_view(), name='report-list'), # Changed name slightly for clarity
    path('reports/upload/', MedicalReportUploadView.as_view(), name='report-upload'),
    # FIX/REFINEMENT: Combine list/detail views for reports
    path('reports/<int:pk>/', MedicalReportDeleteView.as_view(), name='report-delete'), # Uses the <int:pk>/ path for deletion

    # Chat Endpoints
    path('chat/', ChatView.as_view(), name='report-followup-chat'), # Follow-up chat (after analysis)
    # FIX: Changed path to 'chat/query/' to match the frontend URL structure (API_URL + /api/chat/query)
    # It seems your frontend uses /api/chat/query. We need to match that.
    path('chat/query/', GeneralChatView.as_view(), name='general-chat-query'), 

    # Nearby Doctors
    path('doctors/nearby/', NearbyDoctorsView.as_view(), name='nearby-doctors'),
    
    # Authentication
    path('register/', RegisterView.as_view(), name='register'),
    path('verify-otp/', VerifyOTPView.as_view(), name='verify-otp'),
    path('password-reset/', PasswordResetRequestView.as_view(), name='password-reset-request'),
    path('password-reset-confirm/<uidb64>/<token>/', PasswordResetConfirmView.as_view(), name='password-reset-confirm'),
    
]
