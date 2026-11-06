/*
  MG995 - Hieu chinh offset + doc feedback ADC cho SERVO 3 tren ESP32
  Phien ban: Noi suy 2 doan (0-90-180) de sua loi phi tuyen tinh cua bien tro.

  Theo so do nguyen ly:
    Servo 3 Signal (SERVO3) -> GPIO25 (Pin 10)
    Feedback (FB3 / RFB3)   -> GPIO39 (SENSOR_VN / Pin 5)
    Nguon Servo 3: 5.2V qua tu loc C18 (104)

  Serial Monitor: 115200 baud, chon "Newline"
*/

#include <ESP32Servo.h>
#include <Preferences.h>

// ================== CAU HINH THEO SO DO ==================
const int SERVO_PIN = 25;  // IO25 (Pin 10) tren ESP32 dieu khien SERVO3
const int FB_PIN    = 39;  // SENSOR_VN (Pin 5) doc feedback RFB3

// Ty so chia ap = (R1 + R2) / R2 (Vi du: 47k + 47k -> 2.0f)
const float FB_DIVIDER_RATIO = 2.0f;

// Hanh trinh servo (do)
const int SERVO_RANGE_DEG = 180;

const int PULSE_LIMIT_LOW  = 400;    // Gioi han an toan khi chinh offset (us)
const int PULSE_LIMIT_HIGH = 2600;
const int PULSE_STEP       = 5;      // Moi lan bam q/e/z/c doi bao nhieu us

const int DEFAULT_PULSE_MIN = 500;
const int DEFAULT_PULSE_MAX = 2500;

// Nguong bao hoa ADC (mV tai chan ESP32)
const int ADC_SAT_MV = 3100;

// So chu so thap phan khi lam tron fb va err
const int FB_DECIMALS = 0;

// ================== BIEN TOAN CUC ==================
Servo servo;
Preferences prefs;

int pulseMin = DEFAULT_PULSE_MIN;    // us ung voi 0 do
int pulseMax = DEFAULT_PULSE_MAX;    // us ung voi goc cuoi (SERVO_RANGE_DEG)
int adc0     = 1000;                 // mV wiper tai 0 do (se cap nhat khi bam 'k')
int adcMid   = 3342;                 // mV wiper tai 90 do
int adcMax   = 6184;                 // mV wiper tai 180 do

int lastPinMv = 0;                   // Dien ap thuc te tai chan IO39 (SENSOR_VN) o lan doc gan nhat
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

float mvToAngle(int mv) {
  if (adcMax == adc0) return 0;
  
  float a = 0;
  bool isReverse = (adcMax < adc0);
  
  if (!isReverse) {
    if (mv <= adcMid) {
      if (adcMid == adc0) return 0;
      a = (mv - adc0) * 90.0f / (adcMid - adc0);
    } else {
      if (adcMax == adcMid) return 90.0f;
      a = 90.0f + (mv - adcMid) * 90.0f / (adcMax - adcMid);
    }
  } else {
    if (mv >= adcMid) {
      if (adcMid == adc0) return 0;
      a = (mv - adc0) * 90.0f / (adcMid - adc0);
    } else {
      if (adcMax == adcMid) return 90.0f;
      a = 90.0f + (mv - adcMid) * 90.0f / (adcMax - adcMid);
    }
  }
  return constrain(a, 0, SERVO_RANGE_DEG);
}

float roundTo(float x, int d) {
  float p = powf(10.0f, d);
  float r = roundf(x * p) / p;
  if (r == 0) r = 0.0f;
  return r;
}

void moveTo(float a, int settleMs = 700) {
  cmdAngle = constrain(a, 0, SERVO_RANGE_DEG);
  servo.writeMicroseconds(angleToUs(cmdAngle));
  delay(settleMs);
}

void saveCal() {
  prefs.begin("mg995_s3", false); // Luu vao namespace rieng cho Servo 3
  prefs.putInt("pMin", pulseMin);
  prefs.putInt("pMax", pulseMax);
  prefs.putInt("a0", adc0);
  prefs.putInt("aMid", adcMid);
  prefs.putInt("aMax", adcMax);
  prefs.end();
  Serial.println(">> Da luu hieu chinh Servo 3 vao flash.");
}

void loadCal() {
  prefs.begin("mg995_s3", true); // Doc namespace cua Servo 3
  pulseMin = prefs.getInt("pMin", pulseMin);
  pulseMax = prefs.getInt("pMax", pulseMax);
  adc0     = prefs.getInt("a0", adc0);
  adcMid   = prefs.getInt("aMid", adcMid);
  adcMax   = prefs.getInt("aMax", adcMax);
  prefs.end();
}

void printCal() {
  Serial.printf("[SERVO 3] pulseMin=%d us | pulseMax=%d us | adc0=%d mV | adcMid=%d mV | adcMax=%d mV (wiper)\n",
                pulseMin, pulseMax, adc0, adcMid, adcMax);
}

void calibrateADC() {
  Serial.println(">> Calib ADC Servo 3: ve 0 do...");
  moveTo(0, 2000);
  int v0 = readFeedbackMv();
  bool sat0 = lastPinMv > ADC_SAT_MV;

  Serial.println(">> Calib ADC Servo 3: ve 90 do...");
  moveTo(90, 1500);
  int vMid = readFeedbackMv();
  bool satMid = lastPinMv > ADC_SAT_MV;

  Serial.printf(">> Ve %d do...\n", SERVO_RANGE_DEG);
  moveTo(SERVO_RANGE_DEG, 2000);
  int vMax = readFeedbackMv();
  bool satMax = lastPinMv > ADC_SAT_MV;

  Serial.printf(">> Do duoc: adc0=%d mV, adcMid=%d mV, adcMax=%d mV (wiper)\n", v0, vMid, vMax);

  if (sat0 || satMid || satMax) {
    Serial.println("!! ADC bao hoa (> 3100 mV tai IO39). Kiem tra cau chia ap roi do lai. KHONG cap nhat.");
  } else if (abs(vMax - v0) < 200) {
    Serial.println("!! Chenh lech < 200 mV, feedback chua noi hoac sai chan IO39. KHONG cap nhat.");
  } else {
    adc0 = v0;
    adcMid = vMid;
    adcMax = vMax;
    Serial.println(">> Calib hoan tat! Bam 'p' de luu vao Flash.");
  }
  moveTo(SERVO_RANGE_DEG / 2.0f, 1000);
}

void sweepTest() {
  Serial.println(">> Quet kiem tra Servo 3 (5 diem):");
  const float R = SERVO_RANGE_DEG;
  const float pts[5] = {0, R * 0.25f, R * 0.5f, R * 0.75f, R};
  for (int i = 0; i < 5; i++) {
    moveTo(pts[i], 1500);
    int mv = readFeedbackMv();
    float fbRaw = mvToAngle(mv);
    float fb    = roundTo(fbRaw, FB_DECIMALS);
    float err   = roundTo(fbRaw - pts[i], FB_DECIMALS);
    Serial.printf("   cmd=%6.1f | wiper=%4d mV | fb=%6.*f | err=%6.*f\n",
                  pts[i], mv, FB_DECIMALS, fb, FB_DECIMALS, err);
  }
  moveTo(R / 2.0f, 800);
}

void printHelp() {
  Serial.println(F(
    "\n=============== LENH CALIB SERVO 3 ===============\n"
    "g<goc> : di toi goc, vd g45\n"
    "0 9 1  : di toi 0 / giua (90) / cuoi (180) hanh trinh\n"
    "q / e  : pulseMin  -5 / +5 us (offset 0 do)\n"
    "z / c  : pulseMax  -5 / +5 us (offset goc cuoi)\n"
    "k      : tu dong do ADC tai 3 diem (0, 90, 180 do)\n"
    "s      : quet 5 diem kiem tra sai so\n"
    "p      : luu cau hinh hien tai vao flash\n"
    "r      : reset pulse ve mac dinh (chua luu)\n"
    "i      : in thong so hien tai\n"
    "h      : tro giup\n"
    "==================================================\n"));
}

void handleCmd(String s) {
  s.trim();
  if (s.length() == 0) return;
  char c = s[0];
  const float R = SERVO_RANGE_DEG;

  switch (c) {
    case 'g': moveTo(s.substring(1).toFloat(), 500); break;
    case '0': moveTo(0);        break;
    case '9': moveTo(R / 2.0f); break;
    case '1': moveTo(R);        break;

    case 'q': pulseMin = max(pulseMin - PULSE_STEP, PULSE_LIMIT_LOW);  moveTo(0, 300); break;
    case 'e': pulseMin = min(pulseMin + PULSE_STEP, pulseMax - 100);   moveTo(0, 300); break;
    case 'z': pulseMax = max(pulseMax - PULSE_STEP, pulseMin + 100);   moveTo(R, 300); break;
    case 'c': pulseMax = min(pulseMax + PULSE_STEP, PULSE_LIMIT_HIGH); moveTo(R, 300); break;

    case 'k': calibrateADC(); break;
    case 's': sweepTest(); break;
    case 'p': saveCal(); break;
    case 'r':
      pulseMin = DEFAULT_PULSE_MIN;
      pulseMax = DEFAULT_PULSE_MAX;
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
  analogSetPinAttenuation(FB_PIN, ADC_11db); // Cau hinh doc ADC toi da ~3.1V tren IO39 (SENSOR_VN)

  loadCal();

  ESP32PWM::allocateTimer(0);
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, PULSE_LIMIT_LOW, PULSE_LIMIT_HIGH);
  moveTo(SERVO_RANGE_DEG / 2.0f, 1200);

  printHelp();
  printCal();
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
    Serial.printf("cmd=%6.1f deg | pulse=%4d us | wiper=%4d mV | fb=%4.*f deg | err=%4.*f%s\n",
                  cmdAngle, angleToUs(cmdAngle), mv, FB_DECIMALS, fb, FB_DECIMALS, err,
                  lastPinMv > ADC_SAT_MV ? "  !! ADC BAO HOA" : "");
  }
}