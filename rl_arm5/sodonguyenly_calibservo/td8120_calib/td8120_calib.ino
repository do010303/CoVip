/*
  TD8120MG (servo 180 do) - Hieu chinh offset + doc feedback ADC cho SERVO 1
  Phien ban: Noi suy 2 doan (0-90-180) de sua loi phi tuyen tinh cua bien tro.

  Theo so do nguyen ly:
    Servo 1 Signal (SERVO1) -> GPIO27 (Pin 12)
    Feedback (FB1 / RFB1)   -> GPIO35 (Pin 7 / ADC1_CH7)
    Nguon Servo 1: 5.2V qua tu loc C14 (104)

  Cau chia ap (FB_DIVIDER_RATIO = 2.0f):
      wiper ---[47k]---+--- GPIO35
                       |
                     [47k]  + tu 100nF song song
                       |
                      GND

  Serial Monitor: 115200 baud, chon "Newline"
*/

#include <ESP32Servo.h>
#include <Preferences.h>

// ================== CAU HINH THEO SO DO NGUYEN LY ==================
const int SERVO_PIN = 27;  // IO27 (Pin 12) tren ESP32 dieu khien SERVO1
const int FB_PIN    = 35;  // IO35 (Pin 7) doc feedback RFB1

// Ty so chia ap = (R1 + R2) / R2. 47k+47k = 2.0 | noi thang = 1.0
const float FB_DIVIDER_RATIO = 2.0f;

// Hanh trinh servo: 180 do
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
int pulseMax = DEFAULT_PULSE_MAX;    // us ung voi 180 do
int adc0     = 1000;                 // mV wiper tai 0 do (se cap nhat khi bam 'k')
int adcMid   = 2500;                 // mV wiper tai 90 do
int adc180   = 4000;                 // mV wiper tai 180 do

int lastPinMv = 0;                   // Dien ap thuc te tai chan IO35
float cmdAngle = SERVO_RANGE_DEG / 2.0f;

// ================== TIEN ICH ==================
// Tra ve dien ap wiper thuc te (da nhan ty so chia ap)
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
  a = constrain(a, 0.0f, 180.0f);
  return pulseMin + (long)((pulseMax - pulseMin) * a / 180.0f);
}

// Noi suy 2 doan de xu ly bien tro phi tuyen tinh
float mvToAngle(int mv) {
  if (adc180 == adc0) return 0;
  
  float a = 0;
  bool isReverse = (adc180 < adc0); // Kiem tra feedback thuan hay dao chieu
  
  if (!isReverse) {
    if (mv <= adcMid) {
      if (adcMid == adc0) return 0;
      a = (mv - adc0) * 90.0f / (adcMid - adc0);
    } else {
      if (adc180 == adcMid) return 90.0f;
      a = 90.0f + (mv - adcMid) * 90.0f / (adc180 - adcMid);
    }
  } else {
    if (mv >= adcMid) {
      if (adcMid == adc0) return 0;
      a = (mv - adc0) * 90.0f / (adcMid - adc0);
    } else {
      if (adc180 == adcMid) return 90.0f;
      a = 90.0f + (mv - adcMid) * 90.0f / (adc180 - adcMid);
    }
  }
  return constrain(a, 0.0f, 180.0f);
}

// Lam tron x den d chu so thap phan
float roundTo(float x, int d) {
  float p = powf(10.0f, d);
  float r = roundf(x * p) / p;
  if (r == 0) r = 0.0f;
  return r;
}

void moveTo(float a, int settleMs = 700) {
  cmdAngle = constrain(a, 0.0f, 180.0f);
  servo.writeMicroseconds(angleToUs(cmdAngle));
  delay(settleMs);
}

void saveCal() {
  prefs.begin("td8120_s1", false); // Namespace rieng cho Servo 1
  prefs.putInt("pMin", pulseMin);
  prefs.putInt("pMax", pulseMax);
  prefs.putInt("a0", adc0);
  prefs.putInt("aMid", adcMid);
  prefs.putInt("a180", adc180);
  prefs.end();
  Serial.println(">> Da luu hieu chinh Servo 1 vao flash.");
}

void loadCal() {
  prefs.begin("td8120_s1", true);
  pulseMin = prefs.getInt("pMin", pulseMin);
  pulseMax = prefs.getInt("pMax", pulseMax);
  adc0     = prefs.getInt("a0", adc0);
  adcMid   = prefs.getInt("aMid", adcMid);
  adc180   = prefs.getInt("a180", adc180);
  prefs.end();
}

void printCal() {
  Serial.printf("[SERVO 1] pulseMin=%d us | pulseMax=%d us | adc0=%d mV | adcMid=%d mV | adc180=%d mV (wiper)\n",
                pulseMin, pulseMax, adc0, adcMid, adc180);
}

// Tu dong do lay mau ADC tai 3 diem (0, 90, 180)
void calibrateADC() {
  Serial.println(">> Calib ADC Servo 1: ve 0 do...");
  moveTo(0, 2000);
  int v0 = readFeedbackMv();
  bool sat0 = lastPinMv > ADC_SAT_MV;

  Serial.println(">> Calib ADC Servo 1: ve 90 do...");
  moveTo(90, 1500);
  int vMid = readFeedbackMv();
  bool satMid = lastPinMv > ADC_SAT_MV;

  Serial.println(">> Ve 180 do...");
  moveTo(180, 2000);
  int v180 = readFeedbackMv();
  bool satMax = lastPinMv > ADC_SAT_MV;

  Serial.printf(">> Do duoc: adc0=%d mV, adcMid=%d mV, adc180=%d mV (wiper)\n", v0, vMid, v180);

  if (sat0 || satMid || satMax) {
    Serial.println("!! ADC bao hoa (> 3100 mV tai IO35). Kiem tra cau chia ap roi do lai. KHONG cap nhat.");
  } else if (abs(v180 - v0) < 200) {
    Serial.println("!! Chenh lech < 200 mV, feedback chua noi hoac sai chan IO35. KHONG cap nhat.");
  } else {
    adc0 = v0;
    adcMid = vMid;
    adc180 = v180;
    if (adc180 < adc0) Serial.println(">> Feedback dao chieu (goc tang -> dien ap giam), da tu xu ly.");
    Serial.println(">> Calib hoan tat! Bam 'p' de luu vao Flash.");
  }
  moveTo(90, 1000);
}

// Quet 5 diem de kiem tra sai so
void sweepTest() {
  Serial.println(">> Quet kiem tra Servo 1 (5 diem):");
  const float pts[5] = {0.0f, 45.0f, 90.0f, 135.0f, 180.0f};
  for (int i = 0; i < 5; i++) {
    moveTo(pts[i], 1500);
    int mv = readFeedbackMv();
    float fbRaw = mvToAngle(mv);
    float fb    = roundTo(fbRaw, FB_DECIMALS);
    float err   = roundTo(fbRaw - pts[i], FB_DECIMALS);
    Serial.printf("   cmd=%5.1f | wiper=%4d mV | fb=%6.*f | err=%6.*f\n",
                  pts[i], mv, FB_DECIMALS, fb, FB_DECIMALS, err);
  }
  moveTo(90, 800);
}

void printHelp() {
  Serial.println(F(
    "\n=============== LENH CALIB SERVO 1 (TD8120MG) ===============\n"
    "g<goc> : di toi goc, vd g45\n"
    "0 9 1  : di toi 0 / 90 / 180 do\n"
    "q / e  : pulseMin  -5 / +5 us (offset 0 do)\n"
    "z / c  : pulseMax  -5 / +5 us (offset 180 do)\n"
    "k      : tu do ADC tai 3 diem (0, 90, 180 do)\n"
    "s      : quet 5 diem kiem tra sai so\n"
    "p      : luu cau hinh hien tai vao flash\n"
    "r      : reset ve mac dinh (chua luu)\n"
    "i      : in thong so hien tai\n"
    "h      : tro giup\n"
    "=============================================================\n"));
}

void handleCmd(String s) {
  s.trim();
  if (s.length() == 0) return;
  char c = s[0];

  switch (c) {
    case 'g': moveTo(s.substring(1).toFloat(), 500); break;
    case '0': moveTo(0);   break;
    case '9': moveTo(90);  break;
    case '1': moveTo(180); break;

    case 'q': pulseMin = max(pulseMin - PULSE_STEP, PULSE_LIMIT_LOW);  moveTo(0, 300);   break;
    case 'e': pulseMin = min(pulseMin + PULSE_STEP, pulseMax - 100);   moveTo(0, 300);   break;
    case 'z': pulseMax = max(pulseMax - PULSE_STEP, pulseMin + 100);   moveTo(180, 300); break;
    case 'c': pulseMax = min(pulseMax + PULSE_STEP, PULSE_LIMIT_HIGH); moveTo(180, 300); break;

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
  analogSetPinAttenuation(FB_PIN, ADC_11db); // Cau hinh doc toi da ~3.1V tai chan IO35

  loadCal();

  ESP32PWM::allocateTimer(0);
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, PULSE_LIMIT_LOW, PULSE_LIMIT_HIGH);
  moveTo(90, 1000);

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