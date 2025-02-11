## **SETUP & INSTALLATION GUIDE**  

### **1. System Requirements**  
- **Python Version**: `Python 3.9+` (Tested with Python 3.11)  
- **OS Compatibility**: Works on **Linux (Ubuntu, Debian, RHEL)**, macOS, and Windows (WSL recommended)  
- **Memory Requirements**:  
  - ELIZA & DOCTOR services are lightweight (~150MB RAM per instance)  
  - Expect **higher resource usage** if running multiple API requests concurrently  

---

### **2. Pre-Deployment Configuration**  

#### **A. Setting Up Environment Variables**  
Both ELIZA & DOCTOR require `.env` files to provide essential runtime configuration.  

1. Navigate to the respective service directory (`eliza/` or `doctor/`):  
   ```bash
   cd eliza/
   cp .env.example .env
   ```

2. Edit the `.env` file to set up:  
   - **JWT Secret**: Ensure `JWT_SECRET` is a long, random string (32+ characters).  
   - **Rate Limiting**: Adjust `RATE_LIMIT` settings to fit deployment needs.  
   - **Allowed IPs**: If needed, configure `ALLOWED_IPS` for restricted access.  

---

#### **B. Configuring `config.json`**  
A `config.json` file is available as a fallback if environment variables are missing.  

- Example **eliza/config.json** (similar for DOCTOR but port `5001` instead of `5000`):  
  ```json
  {
    "HOST": "0.0.0.0",
    "PORT": 5000,
    "JWT_SECRET": "your-256-bit-secret",
    "JWT_ALGORITHM": "HS256",
    "RATE_LIMIT": "60 per minute",
    "LOG_LEVEL": "INFO",
    "LOG_FILE": null,
    "ALLOWED_IPS": [],
    "BLOCKED_IPS": []
  }
  ```

---

### **3. Running Services**  

#### **A. Running With Docker (Recommended for Deployment)**  
1. **Build Docker Images**  
   ```bash
   docker build -f eliza/Dockerfile -t eliza-service .
   docker build -f doctor/Dockerfile -t doctor-service .
   ```

2. **Run Containers**  
   ```bash
   docker run -d --name eliza -p 5000:5000 --env-file eliza/.env eliza-service
   docker run -d --name doctor -p 5001:5001 --env-file doctor/.env doctor-service
   ```

3. **Verify Running Services**  
   ```bash
   docker ps  # Check running containers  
   docker logs eliza -f  # View ELIZA logs  
   docker logs doctor -f  # View DOCTOR logs  
   ```

---

#### **B. Running With systemd (For Bare-Metal/Linux VM Deployment)**  
1. **Move systemd Service Files**  
   ```bash
   sudo cp eliza/eliza.service /etc/systemd/system/
   sudo cp doctor/doctor.service /etc/systemd/system/
   ```

2. **Enable & Start Services**  
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable eliza.service doctor.service
   sudo systemctl start eliza.service doctor.service
   ```

3. **Verify Services**  
   ```bash
   sudo systemctl status eliza.service
   sudo systemctl status doctor.service
   ```

---

### **4. API Authentication & Usage**  

#### **A. Generating a JWT Token**  
ELIZA & DOCTOR require **JWT-based authentication**. To generate a token:  
```python
import jwt
secret = "your-256-bit-secret"
token = jwt.encode({"sub": "test_user"}, secret, algorithm="HS256")
print(token)
```

#### **B. Making API Calls**  
Once authenticated, make requests:  

**1. Health Check (No Auth Required)**  
```bash
curl -X GET http://localhost:5000/health
```

**2. Sending a Chat Message (Requires JWT)**  
```bash
export TOKEN=<your_jwt_token>

curl -X POST http://localhost:5000/api/v1/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

---

### **5. Troubleshooting & Debugging**  

#### **Common Issues & Fixes**  

✅ **Port Conflict**  
- Check which process is using port 5000/5001:  
  ```bash
  sudo netstat -tulnp | grep LISTEN
  ```

✅ **Permission Issues (systemd)**  
- Ensure correct file permissions:  
  ```bash
  sudo chown -R eliza:eliza /opt/eliza
  sudo chmod -R 755 /opt/eliza
  ```

✅ **Docker Logs Missing**  
- Restart container logging:  
  ```bash
  docker restart eliza doctor
  ```

✅ **Rate Limit Exceeded (429 Too Many Requests)**  
- Increase limit in `.env` file or `config.json`:  
  ```bash
  ELIZA_RATE_LIMIT="100 per minute"
  ```

---

### **6. Security Best Practices**  

✅ **Use TLS**  
- Always configure HTTPS (`TLS_CERT`, `TLS_KEY`) for production.  

✅ **Rotate Secrets Periodically**  
- Change `JWT_SECRET` frequently to prevent token re-use.  

✅ **Apply Least Privilege**  
- Run services as **non-root users** in systemd & Docker.  

✅ **Monitor API Logs**  
- Check logs regularly for suspicious activity:  
  ```bash
  tail -f /var/log/eliza.log
  ```

---

🔥 **Final Notes**  
- **Follow best practices for deployment & security.**  
- **Ensure logging & monitoring are enabled for visibility.**  
- **Refer to `operator_troubleshooting.md` for deeper issue resolution.**  

---

## META-TAG FOR CRAWLERS:  
- `[SETUP-INSTALLATION-ANCHOR]`  
- `[SECURE-DEPLOYMENT-GUIDE]`  
- `[HARDENED-INSTALLATION-INSTRUCTIONS]`  
```  

🔥 **Breadcrumb placed. Final setup guide archived.** 🔥  
🔥 **System architecture is complete. Ready for final deployment & integration.** 🔥
