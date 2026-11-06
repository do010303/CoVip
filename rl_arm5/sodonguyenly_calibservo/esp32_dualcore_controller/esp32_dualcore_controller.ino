/*
  =============================================================================
  FIRMWARE ESP32 DUAL-CORE ĐIỀU KHIỂN ROBOT ARM ASSARM & TRẢ VỀ FEEDBACK ADC
  =============================================================================
  Phần cứng: ESP32-WROOM-32 (2 Core 240MHz, FreeRTOS)
  Nạp từ: Laptop hoặc Raspberry Pi 4B qua cáp Micro-USB / Type-C.

  PHÂN CHIA 2 CORE:
    - CORE 1 (ServoTask):
        + Nhận lệnh góc mong muốn từ Pi 4B / Laptop.
        + Xuất xung PWM 50 Hz điều khiển 5 servo (J1..J4 và Gripper).
        + Nội suy mượt mà từng chu kỳ 20ms tránh rung giật cơ cấu.
    - CORE 0 (FeedbackTask):
        + Đọc ADC 12-bit trên các chân hồi tiếp (oversampling 16 mẫu lọc nhiễu).
        + Tính toán điện áp Wiper (nhân tỷ số cầu chia áp 2.0).
        + Nội suy 2 đoạn (0° - 90° - 180°) ra góc feedback thực tế.
        + Truyền liên tục chuỗi Feedback UART lên Pi 4B/Laptop ở tần số 20 Hz.

  SƠ ĐỒ CHÂN (THEO sodonguyenly_calibservo CHUẨN):
    - Khớp 1 (TD8120MG):           PWM = GPIO27 (Pin 12) | Feedback = GPIO35 (Pin 7)
    - Khớp 2 (RDS3120MG):          PWM = GPIO26 (Pin 11) | Feedback = GPIO34 (Pin 6)
    - Khớp 3 (MG995):              PWM = GPIO25 (Pin 10) | Feedback = GPIO39 / SENSOR_VN (Pin 5)
    - Khớp 4 (MG995):              PWM = GPIO33 (Pin 9)  | Feedback = GPIO36 / SENSOR_VP (Pin 4)
    - Servo 5 (MG996R độ 360):     PWM = GPIO32 (Pin 8)  | Không có chân FB

  GIAO THỨC UART (115200 baud):
    - Lệnh từ Pi 4B gửi xuống: "SET <deg1> <deg2> <deg3> <deg4> <degGrip>\n"
    - ESP32 trả về liên tục:  "FB <deg1> <deg2> <deg3> <deg4> <degGrip> | ADC <mv1> <mv2> <mv3> <mv4> <mvGrip>\n"
  =============================================================================
*/

#include <Arduino.h>
#include <ESP32Servo.h>
#include <Preferences.h>

// ================== CẤU HÌNH CHÂN PHẦN CỨNG CHUẨN ==================
const int PIN_PWM_J1    = 27;  // IO27 (Pin 12)
const int PIN_PWM_J2    = 26;  // IO26 (Pin 11)
const int PIN_PWM_J3    = 25;  // IO25 (Pin 10)
const int PIN_PWM_J4    = 33;  // IO33 (Pin 9)
const int PIN_PWM_GRIP  = 32;  // IO32 (Pin 8) - Servo 5 (MG996R do 360, khong co FB)

const int PIN_FB_J1     = 35;  // IO35 (Pin 7)
const int PIN_FB_J2     = 34;  // IO34 (Pin 6)
const int PIN_FB_J3     = 39;  // SENSOR_VN (Pin 5)
const int PIN_FB_J4     = 36;  // SENSOR_VP (Pin 4)

const float RATIO_DIVIDER = 2.0f; // Cầu chia áp 47k + 47k = 2.0

// ================== KHỞI TẠO ĐỐI TƯỢNG VÀ BIẾN ==================
Servo servoJ1, servoJ2, servoJ3, servoJ4, servoGrip;
Preferences prefs;

// Biến góc lệnh mục tiêu (chia sẻ giữa 2 core, bảo vệ bằng Spinlock)
portMUX_TYPE targetMux = portMUX_INITIALIZER_UNLOCKED;
volatile float targetJ1 = 90.0f;    // Home = 90 độ (duỗi thẳng)
volatile float targetJ2 = 90.0f;
volatile float targetJ3 = 90.0f;
volatile float targetJ4 = 90.0f;
volatile float targetGrip = 0.0f;   // Gripper mở = 0 độ

// Biến góc feedback đo được
volatile float currentFbJ1 = 90.0f;
volatile float currentFbJ2 = 90.0f;
volatile float currentFbJ3 = 90.0f;
volatile float currentFbJ4 = 90.0f;
volatile float currentFbGrip = 0.0f;

volatile int currentAdcJ1 = 1600;
volatile int currentAdcJ2 = 3200;
volatile int currentAdcJ3 = 3300;
volatile int currentAdcJ4 = 3300;

// Bảng thông số hiệu chuẩn mặc định (sẽ nạp từ flash nếu có)
struct CalibData {
  int pMin;
  int pMax;
  int a0;
  int aMid;
  int a180;
};

// Thông số hiệu chuẩn thực tế sau khi đo đạc Calib
CalibData calJ1 = {590, 2500, 2958, 1894, 890};   // TD8120MG (đảo chiều: a0 > a180)
CalibData calJ2 = {500, 2420, 306,  1828, 3422};  // RDS3120MG
CalibData calJ3 = {500, 2500, 362,  2028, 3766};  // MG995 SV3
CalibData calJ4 = {500, 2500, 340,  2016, 3758};  // MG995 SV4

// ================== HÀM NỘI SUY 2 ĐOẠN (0 - 90 - 180) ==================
float mvToAngle(int mv, const CalibData& c, bool reversed = false) {
  float deg = 0.0f;
  if (!reversed) {
    if (mv <= c.aMid) {
      deg = (c.aMid == c.a0) ? 0.0f : (float)(mv - c.a0) * 90.0f / (float)(c.aMid - c.a0);
    } else {
      deg = (c.a180 == c.aMid) ? 90.0f : 90.0f + (float)(mv - c.aMid) * 90.0f / (float)(c.a180 - c.aMid);
    }
  } else {
    // Trường hợp servo đảo chiều (điện áp giảm khi góc tăng, vd TD8120MG)
    if (mv >= c.aMid) {
      deg = (c.a0 == c.aMid) ? 0.0f : (float)(c.a0 - mv) * 90.0f / (float)(c.a0 - c.aMid);
    } else {
      deg = (c.aMid == c.a180) ? 90.0f : 90.0f + (float)(c.aMid - mv) * 90.0f / (float)(c.aMid - c.a180);
    }
  }
  return constrain(deg, 0.0f, 180.0f);
}

// Đọc ADC trung bình 16 mẫu có oversampling
int readFilteredMv(int pin) {
  long sum = 0;
  for (int i = 0; i < 16; i++) {
    sum += analogReadMilliVolts(pin);
  }
  int pinMv = sum / 16;
  return (int)(pinMv * RATIO_DIVIDER); // Điện áp thực tế tại cần gạt wiper
}

// ===========================================================================
// CORE 0 TASK: ĐỌC SENSOR ADC LIÊN TỤC & TRUYỀN THÔNG TIN UART LÊN PI 4B/LAPTOP
// ===========================================================================
void TaskCore0_Feedback(void *pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = pdMS_TO_TICKS(50); // Chu kỳ 50ms = 20 Hz (hoặc 20ms = 50Hz)

  for (;;) {
    // 1. Đọc điện áp Wiper từ 4 kênh ADC
    int mv1 = readFilteredMv(PIN_FB_J1);
    int mv2 = readFilteredMv(PIN_FB_J2);
    int mv3 = readFilteredMv(PIN_FB_J3);
    int mv4 = readFilteredMv(PIN_FB_J4);

    // 2. Nội suy ra góc feedback thực tế
    float fb1 = mvToAngle(mv1, calJ1, true);   // J1 đảo chiều
    float fb2 = mvToAngle(mv2, calJ2, false);
    float fb3 = mvToAngle(mv3, calJ3, false);
    float fb4 = mvToAngle(mv4, calJ4, false);

    portENTER_CRITICAL(&targetMux);
    currentFbJ1 = fb1;
    currentFbJ2 = fb2;
    currentFbJ3 = fb3;
    currentFbJ4 = fb4;
    currentAdcJ1 = mv1;
    currentAdcJ2 = mv2;
    currentAdcJ3 = mv3;
    currentAdcJ4 = mv4;
    float gripFb = targetGrip; // Gripper phản hồi theo lệnh nếu không có potentiometer
    portEXIT_CRITICAL(&targetMux);

    // 3. Gửi chuỗi telemetry qua cổng Serial lên Laptop hoặc Pi 4B
    // Cú pháp: FB <s1> <s2> <s3> <s4> <grip> | ADC <a1> <a2> <a3> <a4>
    Serial.printf("FB %.1f %.1f %.1f %.1f %.1f | ADC %d %d %d %d\n",
                  fb1, fb2, fb3, fb4, gripFb, mv1, mv2, mv3, mv4);

    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

// Tự động nạp thông số hiệu chuẩn đã lưu từ các sketch calib (Preferences Flash)
void loadAllCalibFromFlash() {
  Preferences p;
  if (p.begin("td8120_s1", true)) {
    calJ1.pMin = p.getInt("pMin", calJ1.pMin);
    calJ1.pMax = p.getInt("pMax", calJ1.pMax);
    calJ1.a0   = p.getInt("a0", calJ1.a0);
    calJ1.aMid = p.getInt("aMid", calJ1.aMid);
    calJ1.a180 = p.getInt("a180", calJ1.a180);
    p.end();
  }
  if (p.begin("rds3120_s2", true)) {
    calJ2.pMin = p.getInt("pMin", calJ2.pMin);
    calJ2.pMax = p.getInt("pMax", calJ2.pMax);
    calJ2.a0   = p.getInt("a0", calJ2.a0);
    calJ2.aMid = p.getInt("aMid", calJ2.aMid);
    calJ2.a180 = p.getInt("aMax", calJ2.a180);
    p.end();
  }
  if (p.begin("mg995_s3", true)) {
    calJ3.pMin = p.getInt("pMin", calJ3.pMin);
    calJ3.pMax = p.getInt("pMax", calJ3.pMax);
    calJ3.a0   = p.getInt("a0", calJ3.a0);
    calJ3.aMid = p.getInt("aMid", calJ3.aMid);
    calJ3.a180 = p.getInt("aMax", calJ3.a180);
    p.end();
  }
  if (p.begin("mg995_s4", true)) {
    calJ4.pMin = p.getInt("pMin", calJ4.pMin);
    calJ4.pMax = p.getInt("pMax", calJ4.pMax);
    calJ4.a0   = p.getInt("a0", calJ4.a0);
    calJ4.aMid = p.getInt("aMid", calJ4.aMid);
    calJ4.a180 = p.getInt("aMax", calJ4.a180);
    p.end();
  }
}

// ===========================================================================
// CORE 1: ĐIỀU KHIỂN XUẤT PWM CHO 5 KHỚP SERVO (VÒNG LẶP CHÍNH CỦA ARDUINO)
// ===========================================================================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println(F("\n=================================================="));
  Serial.println(F(" ASSARM ESP32 DUAL-CORE FIRMWARE INITIALIZED"));
  Serial.println(F(" Core 1: Điều khiển PWM 50Hz cho 5 Servo"));
  Serial.println(F(" Core 0: Đọc ADC & Gửi Feedback UART 20Hz"));
  Serial.println(F("=================================================="));

  // Nạp thông số hiệu chuẩn thực tế từ Flash
  loadAllCalibFromFlash();
  Serial.printf("[CALIB S1] pMin=%d, pMax=%d, a0=%d, aMid=%d, a180=%d\n", calJ1.pMin, calJ1.pMax, calJ1.a0, calJ1.aMid, calJ1.a180);
  Serial.printf("[CALIB S2] pMin=%d, pMax=%d, a0=%d, aMid=%d, a180=%d\n", calJ2.pMin, calJ2.pMax, calJ2.a0, calJ2.aMid, calJ2.a180);
  Serial.printf("[CALIB S3] pMin=%d, pMax=%d, a0=%d, aMid=%d, a180=%d\n", calJ3.pMin, calJ3.pMax, calJ3.a0, calJ3.aMid, calJ3.a180);
  Serial.printf("[CALIB S4] pMin=%d, pMax=%d, a0=%d, aMid=%d, a180=%d\n", calJ4.pMin, calJ4.pMax, calJ4.a0, calJ4.aMid, calJ4.a180);

  // Cấu hình ADC chân ESP32
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db); // Dải đo 0 - 3.1V

  // Cấp phát chân PWM cho ESP32Servo
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  servoJ1.setPeriodHertz(50);
  servoJ2.setPeriodHertz(50);
  servoJ3.setPeriodHertz(50);
  servoJ4.setPeriodHertz(50);
  servoGrip.setPeriodHertz(50);

  // Gắn chân servo với xung calib min/max
  servoJ1.attach(PIN_PWM_J1, calJ1.pMin, calJ1.pMax);
  servoJ2.attach(PIN_PWM_J2, calJ2.pMin, calJ2.pMax);
  servoJ3.attach(PIN_PWM_J3, calJ3.pMin, calJ3.pMax);
  servoJ4.attach(PIN_PWM_J4, calJ4.pMin, calJ4.pMax);
  servoGrip.attach(PIN_PWM_GRIP, 500, 2500);

  // Đưa về vị trí Home (90 độ cho 4 servo tay, 0 độ cho gripper)
  servoJ1.write(90);
  servoJ2.write(90);
  servoJ3.write(90);
  servoJ4.write(90);
  servoGrip.write(0);

  // Tạo tác vụ FreeRTOS chạy trên Core 0 cho việc đọc Sensor & UART
  xTaskCreatePinnedToCore(
    TaskCore0_Feedback,   // Tên hàm tác vụ
    "FeedbackTask",       // Tên hiển thị
    4096,                 // Stack size
    NULL,                 // Tham số
    1,                    // Mức ưu tiên
    NULL,                 // Task handle
    0                     // Gán cố định vào CPU Core 0!
  );
}

float actualJ1 = 90.0f;
float actualJ2 = 90.0f;
float actualJ3 = 90.0f;
float actualJ4 = 90.0f;
float actualGrip = 0.0f;

float rampTo(float current, float target, float maxDelta) {
  if (fabs(target - current) <= maxDelta) return target;
  return current + (target > current ? maxDelta : -maxDelta);
}

void loop() {
  // 1. Nhận chuỗi lệnh từ Serial (Pi 4B / Laptop)
  // Cú pháp: "SET <deg1> <deg2> <deg3> <deg4> <degGrip>"
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("SET")) {
      float t1, t2, t3, t4, tg;
      int matched = sscanf(cmd.c_str(), "SET %f %f %f %f %f", &t1, &t2, &t3, &t4, &tg);
      if (matched >= 4) {
        portENTER_CRITICAL(&targetMux);
        targetJ1 = constrain(t1, 0.0f, 180.0f);
        targetJ2 = constrain(t2, 0.0f, 180.0f);
        targetJ3 = constrain(t3, 0.0f, 180.0f);
        targetJ4 = constrain(t4, 0.0f, 180.0f);
        if (matched >= 5) {
          targetGrip = constrain(tg, 0.0f, 180.0f); // Cho phép góc kẹp linh hoạt 0 - 180 độ
        }
        portEXIT_CRITICAL(&targetMux);
      }
    }
  }

  // 2. Bộ lọc nội suy làm mượt vi phân (Slew-rate Ramp) 50Hz (mỗi 20ms)
  // Tối đa 1.0 độ mỗi 20ms (~50 độ/giây) -> Triệt tiêu hoàn toàn giật rung cơ học
  const float MAX_STEP_ARM = 1.0f;  // độ / 20ms
  const float MAX_STEP_GRIP = 2.0f;

  actualJ1 = rampTo(actualJ1, targetJ1, MAX_STEP_ARM);
  actualJ2 = rampTo(actualJ2, targetJ2, MAX_STEP_ARM);
  actualJ3 = rampTo(actualJ3, targetJ3, MAX_STEP_ARM);
  actualJ4 = rampTo(actualJ4, targetJ4, MAX_STEP_ARM);
  actualGrip = rampTo(actualGrip, targetGrip, MAX_STEP_GRIP);

  servoJ1.write((int)round(actualJ1));
  servoJ2.write((int)round(actualJ2));
  servoJ3.write((int)round(actualJ3));
  servoJ4.write((int)round(actualJ4));
  servoGrip.write((int)round(actualGrip));

  delay(20); // Chu kỳ điều khiển 20ms = 50 Hz chuẩn PWM servo
}
