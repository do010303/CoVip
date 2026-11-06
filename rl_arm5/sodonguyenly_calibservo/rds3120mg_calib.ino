/*
  RDS3120MG (servo 20kg, metal gear) - BAN 180 DO
  Hieu chinh offset + doc feedback ADC cho SERVO 2 tren ESP32
  Noi suy 2 doan (0 - 90 - 180)

  Theo so do nguyen ly:
    Servo 2 Signal (SERVO2) -> GPIO26
    Feedback (FB2 / RFB2)   -> GPIO34 (ADC1_CH6)
    Nguon Servo 2: 5.2V qua tu loc C15 (104)

  Cau chia ap (FB_DIVIDER_RATIO = 2.0):
      wiper ---[47k]---+--- GPIO34
                       |
                     [47k]  + tu 100nF song song
                       |
                      GND

  Serial Monitor: 115200 baud, chon "Newline"
*/

#include <ESP32Servo.h>
#include <Preferences.h>

// ================== CAU HINH THEO SO DO NGUYEN LY ==================
const int SERVO_PIN = 26;  // IO26 (Pin 11) dieu khien SERVO2
const int FB_PIN    = 34;  // IO34 (Pin 6) doc feedback RFB2

// Ty so chia ap = (R1 + R2) / R2. 47k+47k = 2.0 | 47k+22k = 3.14 | noi thang = 1.0
const float FB_DIVIDER_RATIO = 2.0f;

// Hanh trinh servo: ban 180 do
const int SERVO_RANGE_DEG = 180;

const int PULSE_LIMIT_LOW  = 400;    // gioi han an toan khi chinh offset (us)
const int PULSE_LIMIT_HIGH = 2600;
const int PULSE_STEP       = 5;      // moi lan bam q/e/z/c doi bao nhieu us

const int DEFAULT_PULSE_MIN = 500;   // thong so mac dinh: 500-2500 us
const int DEFAULT_PULSE_MAX = 2500;

// Nguong bao hoa ADC (mV tai chan ESP32)
const int ADC_SAT_MV = 3100;

// So chu so thap phan khi lam tron fb va err (0 = so nguyen, 1 = 1 chu so le)
const int FB_DECIMALS = 0;

// ================== BIEN TOAN CUC ==================
Servo servo;
Preferences prefs;

int pulseMin = DEFAULT_PULSE_MIN;    // us ung voi 0 do
int pulseMax = DEFAULT_PULSE_MAX;    // us ung voi 180 do

int adc0     = 1100;                 // mV wiper tai 0 do (se ghi de khi bam 'k')
int adcMid   = 3278;                 // mV wiper tai 90 do
int adcMax   = 5400;                 // mV wiper tai 180 do

bool calibrated = false;             // da chay 'k' hoac nap tu flash chua
int lastPinMv = 0;                   // dien ap thuc te tai chan IO34
float cmdAngle = SERVO_RANGE_DEG / 2.0f;

// ================== TIEN ICH ==================
int readFeedbackMv() {
  long sum = 0;
  const int N = 32;
  for (int i = 0; i < N; i++) {
    sum += analogReadMilliVolts(FB_PIN);
    delay(2);
  }
  lastPinMv = sum / N;
  return (int)(lastPinMv * FB_DIVIDER_RATIO);
}

int angleToUs(float a) {
  a = constrain(a, 0, SERVO_RANGE_DEG);
  return pulseMin + (long)((pulseMax - pulseMin) * a / SERVO_RANGE_DEG);
}

// Noi suy 2 doan (0 -> 90 -> 180), tu dong xu ly khi feedback dao chieu
float mvToAngle(int mv) {
  if (adcMax == adc0) return 0;

  const float halfDeg = SERVO_RANGE_DEG / 2.0f;
  const bool isReverse = (adcMax < adc0);
  const bool inFirstHalf = isReverse ? (mv >= adcMid) : (mv <= adcMid);

  float a;
  if (inFirstHalf) {
    if (adcMid == adc0) return 0;
    a = (mv - adc0) * halfDeg / (adcMid - adc0);
  } else {
    if (adcMax == adcMid) return halfDeg;
    a = halfDeg + (mv - adcMid) * halfDeg / (adcMax - adcMid);
  }
  return constrain(a, 0, SERVO_RANGE_DEG);
}

float roundTo(float x, int d) {
  float p = powf(10.0f, d);
  float r = roundf(x * p) / p;
  if (r == 0) r = 0.0f;
  return r;
}

void moveTo(float a, int settleMs = 800) {
  cmdAngle = constrain(a, 0, SERVO_RANGE_DEG);
  servo.writeMicroseconds(angleToUs(cmdAngle));
  delay(settleMs);
}

void saveCal() {
  prefs.begin("rds3120_s2", false); // Namespace rieng cho Servo 2
  prefs.putInt("pMin", pulseMin);
  prefs.putInt("pMax", pulseMax);
  prefs.putInt("a0", adc0);
  prefs.putInt("aMid", adcMid);
  prefs.putInt("aMax", adcMax);
  prefs.putFloat("ratio", FB_DIVIDER_RATIO);
  prefs.putBool("cal", calibrated);
  prefs.end();
  Serial.println(">> Da luu hieu chinh Servo 2 vao flash.");
}

void loadCal() {
  prefs.begin("rds3120_s2", true);
  pulseMin   = prefs.getInt("pMin", pulseMin);
  pulseMax   = prefs.getInt("pMax", pulseMax);
  adc0       = prefs.getInt("a0", adc0);
  adcMid     = prefs.getInt("aMid", adcMid);
  adcMax     = prefs.getInt("aMax", adcMax);
  calibrated = prefs.getBool("cal", false);
  float savedRatio = prefs.getFloat("ratio", FB_DIVIDER_RATIO);
  prefs.end();

  if (calibrated && fabsf(savedRatio - FB_DIVIDER_RATIO) > 0.01f) {
    calibrated = false;
    Serial.printf("!! Ty so chia ap da doi (%.2f -> %.2f). Chay lai lenh 'k'.\n",
                  savedRatio, FB_DIVIDER_RATIO);
  }
}

void printCal() {
  Serial.printf("[SERVO 2] pulseMin=%d us | pulseMax=%d us | adc0=%d mV | adcMid=%d mV | adcMax=%d mV (wiper)%s\n",
                pulseMin, pulseMax, adc0, adcMid, adcMax,
                calibrated ? "" : "  [CHUA CALIB]");
}

// Do ADC 3 diem (0, 90, 180)
void calibrateADC() {
  Serial.println(">> Calib ADC Servo 2: ve 0 do...");
  moveTo(0, 2500);
  int v0 = readFeedbackMv();
  bool sat0 = lastPinMv > ADC_SAT_MV;

  Serial.println(">> Calib ADC Servo 2: ve 90 do...");
  moveTo(90, 2000);
  int vMid = readFeedbackMv();
  bool satMid = lastPinMv > ADC_SAT_MV;

  Serial.println(">> Calib ADC Servo 2: ve 180 do...");
  moveTo(180, 2500);
  int vMax = readFeedbackMv();
  bool satMax = lastPinMv > ADC_SAT_MV;

  Serial.printf(">> Do duoc: adc0=%d mV, adcMid=%d mV, adcMax=%d mV (wiper)\n", v0, vMid, vMax);

  bool monotonic = (v0 < vMid && vMid < vMax) || (v0 > vMid && vMid > vMax);

  if (sat0 || satMid || satMax) {
    Serial.println("!! ADC bao hoa (> 3100 mV tai IO34). Tang ty so chia ap (vd 3.14) roi do lai. KHONG cap nhat.");
  } else if (abs(vMax - v0) < 200) {
    Serial.println("!! Chenh lech < 200 mV, feedback chua noi hoac sai chan IO34. KHONG cap nhat.");
  } else if (!monotonic) {
    Serial.println("!! Diem giua khong nam giua 2 dau, do lai. KHONG cap nhat.");
  } else {
    adc0       = v0;
    adcMid     = vMid;
    adcMax     = vMax;
    calibrated = true;
    if (adcMax < adc0) Serial.println(">> Feedback dao chieu (goc tang -> dien ap giam), da tu xu ly.");
    Serial.println(">> Calib thanh cong! Bam 'p' de luu vao flash.");
  }
  moveTo(90, 1200);
}

// Quet 5 diem (0, 45, 90, 135, 180) de kiem tra sai so
void sweepTest() {
  if (!calibrated) Serial.println("!! Chua calib, ket qua chua dung. Chay 'k' truoc.");
  Serial.println(">> Quet kiem tra Servo 2 (5 diem):");
  const float pts[5] = {0.0f, 45.0f, 90.0f, 135.0f, 180.0f};
  for (int i = 0; i < 5; i++) {
    moveTo(pts[i], 1800);
    int mv = readFeedbackMv();
    float fbRaw = mvToAngle(mv);
    float fb    = roundTo(fbRaw, FB_DECIMALS);
    float err   = roundTo(fbRaw - pts[i], FB_DECIMALS);
    Serial.printf("   cmd=%6.1f | wiper=%4d mV | fb=%6.*f | err=%6.*f\n",
                  pts[i], mv, FB_DECIMALS, fb, FB_DECIMALS, err);
  }
  moveTo(90, 1000);
}

void printHelp() {
  Serial.println(F(
    "\n=============== LENH CALIB SERVO 2 (RDS3120MG) ===============\n"
    "g<goc> : di toi goc, vd g45\n"
    "0 9 1  : di toi 0 / 90 / 180 do\n"
    "q / e  : pulseMin  -5 / +5 us (offset 0 do)\n"
    "z / c  : pulseMax  -5 / +5 us (offset 180 do)\n"
    "k      : tu do ADC tai 3 diem (0, 90, 180)\n"
    "s      : quet 5 diem kiem tra sai so\n"
    "p      : luu vao flash\n"
    "r      : reset pulse ve mac dinh (chua luu)\n"
    "i      : in thong so hien tai\n"
    "h      : tro giup\n"
    "==============================================================\n"));
}

void handleCmd(String s) {
  s.trim();
  if (s.length() == 0) return;
  char c = s[0];

  switch (c) {
    case 'g': moveTo(s.substring(1).toFloat(), 600); break;
    case '0': moveTo(0);   break;
    case '9': moveTo(90);  break;
    case '1': moveTo(180); break;

    case 'q': pulseMin = max(pulseMin - PULSE_STEP, PULSE_LIMIT_LOW);  moveTo(0, 350);   calibrated = false; break;
    case 'e': pulseMin = min(pulseMin + PULSE_STEP, pulseMax - 100);   moveTo(0, 350);   calibrated = false; break;
    case 'z': pulseMax = max(pulseMax - PULSE_STEP, pulseMin + 100);   moveTo(180, 350); calibrated = false; break;
    case 'c': pulseMax = min(pulseMax + PULSE_STEP, PULSE_LIMIT_HIGH); moveTo(180, 350); calibrated = false; break;

    case 'k': calibrateADC(); break;
    case 's': sweepTest(); break;
    case 'p':
      if (!calibrated) Serial.println("!! Luu y: chua calib ADC (hoac vua doi pulse). Nen chay 'k' truoc khi luu.");
      saveCal();
      break;
    case 'r':
      pulseMin = DEFAULT_PULSE_MIN;
      pulseMax = DEFAULT_PULSE_MAX;
      calibrated = false;
      Serial.println(">> Da reset pulse ve mac dinh.");
      break;
    case 'i': break;
    case 'h': printHelp(); return;
    default:  Serial.println("Lenh khong hop le. Go h de xem tro giup."); return;
  }
  printCal();
}

// ================== SETUP / LOOP ==================
void setup() {
  Serial.begin(115200);
  delay(300);

  analogReadResolution(12);
  analogSetPinAttenuation(FB_PIN, ADC_11db); // Cau hinh doc ADC toi da ~3.1V tai chan IO34

  loadCal();

  ESP32PWM::allocateTimer(0);
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, PULSE_LIMIT_LOW, PULSE_LIMIT_HIGH);
  moveTo(90, 1500);

  printHelp();
  printCal();
  if (!calibrated) Serial.println("!! Chua calib ADC: gia tri fb/err chi la tam. Chinh pulse roi chay 'k'.");
}

void loop() {
  if (Serial.available()) {
    handleCmd(Serial.readStringUntil('\n'));
  }

  static uint32_t t = 0;
  if (millis() - t > 200) {
    t = millis();
    int mv = readFeedbackMv();
    float fbRaw = mvToAngle(mv);
    float fb    = roundTo(fbRaw, FB_DECIMALS);
    float err   = roundTo(fbRaw - cmdAngle, FB_DECIMALS);
    Serial.printf("cmd=%6.1f deg | pulse=%4d us | wiper=%4d mV | fb=%4.*f deg | err=%4.*f%s%s\n",
                  cmdAngle, angleToUs(cmdAngle), mv, FB_DECIMALS, fb, FB_DECIMALS, err,
                  calibrated ? "" : "  [CHUA CALIB]",
                  lastPinMv > ADC_SAT_MV ? "  !! ADC BAO HOA" : "");
  }
}
