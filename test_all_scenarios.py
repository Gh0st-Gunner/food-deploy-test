import requests
import time
import os

BASE_URL = "http://localhost:10800/api/v1"

def test_health():
    print("TC-00: Checking API Health...")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    print("Health Status:", r.json())
    print("-> TC-00: PASS")

def test_models():
    print("\nTC-08: Listing available classification models...")
    r = requests.get(f"{BASE_URL}/models")
    assert r.status_code == 200, f"Models check failed: {r.status_code}"
    models = r.json()
    print("Available Models:", models)
    # Ensure no Gemini or Ollama models remain
    for m in models:
        name = m.get("name", "")
        assert not name.startswith("gemini:"), f"Found forbidden Gemini model: {name}"
        assert not name.startswith("ollama:"), f"Found forbidden Ollama model: {name}"
    print("-> TC-08 (Models Listing): PASS")

def test_auth_and_user_flows():
    print("\nTC-02: Registering new test user...")
    username = f"testuser_{int(time.time())}"
    email = f"{username}@gmail.com"
    password = "password123"
    
    # 1. Register
    r = requests.post(f"{BASE_URL}/auth/register", json={
        "username": username,
        "email": email,
        "password": password
    })
    assert r.status_code == 200, f"Register failed: {r.status_code} - {r.text}"
    user_data = r.json()
    print("Registration response:", user_data)
    
    # 2. Login with Username (TC-03)
    print("TC-03: Logging in with Username...")
    r = requests.post(f"{BASE_URL}/auth/login", json={
        "username": username,
        "password": password
    })
    assert r.status_code == 200, f"Login by username failed: {r.status_code} - {r.text}"
    login_data = r.json()
    token = login_data["session_token"]
    print("Login successful! Token:", token)
    
    # 3. Login with Email (TC-04)
    print("TC-04: Logging in with Email...")
    r = requests.post(f"{BASE_URL}/auth/login", json={
        "username": email,
        "password": password
    })
    assert r.status_code == 200, f"Login by email failed: {r.status_code} - {r.text}"
    print("Login by email: PASS")
    
    # 4. Forgot Password Flow (TC-05)
    print("TC-05: Forgot password flow...")
    r = requests.post(f"{BASE_URL}/auth/forgot-password", json={"email": email})
    assert r.status_code == 200, f"Forgot password failed: {r.status_code} - {r.text}"
    print("Forgot password request sent successfully.")
    
    # Read the reset code from the DB directly to simulate email retrieval
    # Using SQL query on the PostgreSQL DB
    import subprocess
    cmd = ["docker", "compose", "exec", "db", "psql", "-U", "vnfood", "-d", "vnfood", "-t", "-c", f"SELECT verification_code FROM users WHERE username = '{username}';"]
    code = subprocess.check_output(cmd).decode().strip()
    print(f"Retrieved OTP Reset Code from DB: {code}")
    
    # Reset password
    new_password = "newpassword123"
    r = requests.post(f"{BASE_URL}/auth/reset-password", json={
        "email": email,
        "code": code,
        "new_password": new_password
    })
    assert r.status_code == 200, f"Reset password failed: {r.status_code} - {r.text}"
    print("Password reset successful.")
    
    # Verify login with old password fails
    r = requests.post(f"{BASE_URL}/auth/login", json={
        "username": username,
        "password": password
    })
    assert r.status_code == 401, f"Expected login failure with old password, but got: {r.status_code}"
    
    # Verify login with new password succeeds
    r = requests.post(f"{BASE_URL}/auth/login", json={
        "username": username,
        "password": new_password
    })
    assert r.status_code == 200, "Login with new password failed"
    print("-> TC-02, TC-03, TC-04, TC-05: PASS")
    return token

def test_ai_scanner():
    print("\nTC-09: Running AI Scan (Fast Mode) on banh-mi.jpg...")
    img_path = "c:/Users/Home/Desktop/vn food/test-image/banh-mi.jpg"
    assert os.path.exists(img_path), "banh-mi.jpg missing"
    
    # Submit job
    import base64
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    r = requests.post(f"{BASE_URL}/analyze", json={
        "image_base64": img_b64,
        "mode": "fast",
        "models": ["eff_b0"]
    })
        
    assert r.status_code == 200, f"Scan submit failed: {r.status_code} - {r.text}"
    job_id = r.json()["job_id"]
    print(f"Submitted Fast Scan Job. ID: {job_id}")
    
    # Poll job status
    completed = False
    for _ in range(30):
        r = requests.get(f"{BASE_URL}/jobs/{job_id}")
        assert r.status_code == 200, f"Fetch job failed: {r.status_code}"
        job = r.json()
        status = job["status"]
        print(f"Job Status: {status}")
        if status == "completed":
            completed = True
            result_data = job.get("result", {})
            class_name = result_data.get("class_name")
            confidence = result_data.get("confidence")
            print("Classification Result:", class_name, "Confidence:", confidence)
            assert class_name is not None, "Classification returned null"
            break
        elif status == "failed":
            raise ValueError(f"Job failed with error: {job.get('error')}")
        time.sleep(1)
        
    assert completed, "Job did not complete within timeout"
    print("-> TC-09 (Fast Scan): PASS")

def test_admin_endpoints():
    print("\nTC-13: Testing Admin Panel APIs...")
    # Admin login
    r = requests.post(f"{BASE_URL}/auth/login", json={
        "username": "admin",
        "password": "admin123"
    })
    assert r.status_code == 200, "Admin login failed"
    admin_token = r.json()["session_token"]
    
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    # Fetch admin stats
    r = requests.get(f"{BASE_URL}/admin/stats", headers=headers)
    assert r.status_code == 200, f"Stats failed: {r.status_code} - {r.text}"
    print("Admin Stats:", r.json())
    
    # List admin users
    r = requests.get(f"{BASE_URL}/admin/users", headers=headers)
    assert r.status_code == 200, f"User list failed: {r.status_code} - {r.text}"
    users = r.json()
    print(f"Total registered users found: {len(users)}")
    print("-> TC-13 (Admin Panel): PASS")

def main():
    try:
        test_health()
        test_models()
        test_auth_and_user_flows()
        test_ai_scanner()
        test_admin_endpoints()
        print("\n==============================")
        print("ALL API INTEGRATION TESTS PASSED")
        print("==============================")
    except AssertionError as e:
        print("\n!!! TEST SCENARIO SCENARIOS FAILED !!!")
        print(e)
    except Exception as e:
        print("\n!!! UNEXPECTED TEST ERROR !!!")
        print(e)

if __name__ == "__main__":
    main()
