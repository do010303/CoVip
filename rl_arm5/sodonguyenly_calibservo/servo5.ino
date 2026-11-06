#include <ESP32Servo.h>

// SERVO5 noi vao IO32 theo so do nguyen ly
#define SERVO5_PIN 32

// Dai xung chuan cho MG995 180 do
#define MIN_PULSE 500
#define MAX_PULSE 2500

Servo servo5;
int currentAngle = 0; // Luu vi tri goc hien tai
bool isAttached = false;

// Ham quay muot ma tu vi tri hien tai toi goc moi
void moveToAngle(int targetAngle, int stepDelay = 15) {
  if (!isAttached) {
    servo5.attach(SERVO5_PIN, MIN_PULSE, MAX_PULSE);
    isAttached = true;
  }

  targetAngle = constrain(targetAngle, 0, 180);
  Serial.printf("-> Dang quay tu %d do sang %d do...\n", currentAngle, targetAngle);

  if (targetAngle > currentAngle) {
    for (int pos = currentAngle; pos <= targetAngle; pos++) {
      servo5.write(pos);
      delay(stepDelay);
    }
  } else {
    for (int pos = currentAngle; pos >= targetAngle; pos--) {
      servo5.write(pos);
      delay(stepDelay);
    }
  }

  currentAngle = targetAngle;
  Serial.printf("-> Da toi vi tri: %d do.\n", currentAngle);
}

void printHelp() {
  Serial.println(F(
    "\n================ MENU DIEU KHIEN SERVO 5 (IO32) ================\n"
    " - Nhap so goc truc tiep (0 - 180) roi an ENTER (vi du: 0, 45, 90, 180)\n"
    " - '0' : Di toi 0 do\n"
    " - '9' : Di toi 90 do\n"
    " - '1' : Di toi 180 do\n"
    " - 'd' : Nha servo (detach - tat xung giu dong co de khong bi nong)\n"
    " - 'a' : Bat lai servo ve goc gan nhat\n"
    " - 'h' : In menu huong dan\n"
    "================================================================"));
}

void setup() {
  Serial.begin(115200);
  delay(500);

  ESP32PWM::allocateTimer(0);
  servo5.setPeriodHertz(50);
  servo5.attach(SERVO5_PIN, MIN_PULSE, MAX_PULSE);
  isAttached = true;

  // Dua servo ve moc 0 do khoi tao
  servo5.write(currentAngle);
  delay(1000);

  printHelp();
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() == 0) return;

    if (input.equalsIgnoreCase("d")) {
      servo5.detach();
      isAttached = false;
      Serial.println(">> Da nha servo (Detach).");
    } 
    else if (input.equalsIgnoreCase("a")) {
      servo5.attach(SERVO5_PIN, MIN_PULSE, MAX_PULSE);
      isAttached = true;
      servo5.write(currentAngle);
      Serial.printf(">> Da bat lai servo tai vi tri: %d do.\n", currentAngle);
    } 
    else if (input.equalsIgnoreCase("h")) {
      printHelp();
    } 
    else if (input == "0") {
      moveToAngle(0);
    } 
    else if (input == "9") {
      moveToAngle(90);
    } 
    else if (input == "1") {
      moveToAngle(180);
    } 
    else {
      // Kiem tra neu nguoi dung nhap so goc
      int target = input.toInt();
      if (target >= 0 && target <= 180) {
        moveToAngle(target);
      } else {
        Serial.println("!! Goc khong hop le! Vui long nhap so tu 0 den 180.");
      }
    }
  }
}