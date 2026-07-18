// ===================================================
//  자율주행 펌웨어 v4.0 — "가장 조향 잘되는 버전" + 1바이트 시리얼 프로토콜
//
//  베이스: 유격 대응 수동조종 코드 (3핀 제어, 방향별 중립 유격 보정).
//  조향 로직/핀맵/극성/캘리브레이션은 그 버전 그대로 유지하고,
//  키보드(w/a/s/d) 입력 대신 맥의 1바이트 프로토콜을 받도록 교체:
//    비트7=1 조향: v=(b&0x7F) 0..127, 64=중립. 좌우 구간별 map으로 pot 목표 산출
//                  (맥은 -1..1 정규화 조향만 보냄, pot 캘리브레이션은 펌웨어 전담)
//    비트7=0 모터: 0x00/0x40 정지, 0x01~0x3F 전진 b*4, 0x41~0x7F 후진
//  + 워치독(400ms 무수신 시 정지·중립) — 맥 serial_driver.py의
//    150ms 하트비트와 세트로 동작
//
//  조향은 좌우 완전 대칭 P제어 + 데드밴드. (방향별 유격 오프셋은
//  주행 중 상시 우편향을 유발해서 제거함 — 2026-07-11)
// ===================================================

// ==========================================
// 1. 핀 설정 (3핀 제어)
// ==========================================

// [핸들 / 조향 모터]
const int int2_ST = 2; // 노란색
const int int1_ST = 3; // 주황색
const int pwm_ST  = 4; // 초록색 (PWM)
const int potPin  = A0; // 핸들 가변저항

// [운전석 뒷바퀴]
const int int2_LR = 5; // 보라색
const int int1_LR = 6; // 회색
const int pwm_LR  = 7; // 노란색 (PWM)

// [조수석 뒷바퀴]
const int int2_RR = 8; // 초록색
const int int1_RR = 9; // 파란색
const int pwm_RR  = 10; // 파란색 (PWM)

// ==========================================
// 2. 조향 제어 상수 (방향별 중립 유격 보정, calib.json pot_* 와 일치)
// ==========================================
const int STEER_LEFT    = 652;  // 좌측 최대
const int STEER_RIGHT   = 514;  // 우측 최대
const int STEER_NEUTRAL = 582;  // 이론상 정중앙

const int STEER_DEADBAND = 8;   // 오차 허용 범위 (좌우 대칭)
const int MIN_SPEED = 70;       // 조향 모터 최소 PWM
const int MAX_SPEED = 205;      // 조향 모터 최대 PWM
const float STEER_KP = 3.5;     // P 제어 비례 계수

// ==========================================
// 3. 제어 변수
// ==========================================
const uint32_t WATCHDOG_MS = 400;
uint32_t lastRxMs = 0;
bool watchdogTripped = false;

int targetSteering = STEER_NEUTRAL;

void setup() {
  pinMode(int1_ST, OUTPUT); pinMode(int2_ST, OUTPUT); pinMode(pwm_ST, OUTPUT);
  pinMode(int1_LR, OUTPUT); pinMode(int2_LR, OUTPUT); pinMode(pwm_LR, OUTPUT);
  pinMode(int1_RR, OUTPUT); pinMode(int2_RR, OUTPUT); pinMode(pwm_RR, OUTPUT);

  Serial.begin(115200);   // calib.json "baud" 와 일치

  stopDriving();
  lastRxMs = millis();
}

void loop() {
  // 1. 맥 명령 수신 (1바이트 프로토콜)
  while (Serial.available() > 0) {
    processByte(Serial.read());
  }

  // 2. 워치독: 통신 두절 시 모터 정지 + 조향 중립
  if (!watchdogTripped && (millis() - lastRxMs > WATCHDOG_MS)) {
    stopDriving();
    targetSteering = STEER_NEUTRAL;
    watchdogTripped = true;
  }

  // 3. 실시간 조향 핸들 제어
  updateSteering();
}

// ===== 1바이트 명령 처리 =====
void processByte(uint8_t b) {
  lastRxMs = millis();
  watchdogTripped = false;

  if (b & 0x80) {
    // ── 조향: v=0..127 (64=중립), pot 변환은 전적으로 펌웨어 담당 ──
    // 중립이 좌우 중간이 아니어도(링키지 비대칭) 구간별 매핑으로 정확히 대응
    int v = b & 0x7F;
    if (v >= 64) {
      targetSteering = map(v, 64, 127, STEER_NEUTRAL, STEER_LEFT);
    } else {
      targetSteering = map(v, 0, 64, STEER_RIGHT, STEER_NEUTRAL);
    }
  } else {
    // ── 뒷바퀴 모터 ──
    if (b == 0x00 || b == 0x40) {
      stopDriving();
    } else if (b <= 0x3F) {
      driveForward((int)b * 4);
    } else {
      driveBackward((int)(b & 0x3F) * 4);
    }
  }
}

// ===== 뒷바퀴 구동 (좌우 극성 반대인 것에 주의 — 수동 버전 실측 그대로) =====
void driveForward(int pwm) {
  digitalWrite(int1_LR, HIGH); digitalWrite(int2_LR, LOW);  analogWrite(pwm_LR, pwm);
  digitalWrite(int1_RR, LOW);  digitalWrite(int2_RR, HIGH); analogWrite(pwm_RR, pwm);
}

void driveBackward(int pwm) {
  digitalWrite(int1_LR, LOW);  digitalWrite(int2_LR, HIGH); analogWrite(pwm_LR, pwm);
  digitalWrite(int1_RR, HIGH); digitalWrite(int2_RR, LOW);  analogWrite(pwm_RR, pwm);
}

void stopDriving() {
  analogWrite(pwm_LR, 0);
  analogWrite(pwm_RR, 0);
  digitalWrite(int1_LR, LOW); digitalWrite(int2_LR, LOW);
  digitalWrite(int1_RR, LOW); digitalWrite(int2_RR, LOW);
}

// ===== 조향 모터를 목표값으로 실시간 구동 (좌우 대칭 P제어) =====
void updateSteering() {
  int currentPot = analogRead(potPin);

  int error = targetSteering - currentPot;
  if (abs(error) <= STEER_DEADBAND) {
    // 목표 도달 → 정지
    analogWrite(pwm_ST, 0);
    digitalWrite(int1_ST, LOW);
    digitalWrite(int2_ST, LOW);
    return;
  }

  // P 제어 기반 속도 계산 및 제한
  int speed = abs(error) * STEER_KP;
  if (speed > MAX_SPEED) speed = MAX_SPEED;
  if (speed < MIN_SPEED) speed = MIN_SPEED;

  // 우측(514) < 중립(582) < 좌측(652): error>0 = 왼쪽으로
  if (error > 0) {
    digitalWrite(int1_ST, HIGH);
    digitalWrite(int2_ST, LOW);
    analogWrite(pwm_ST, speed);
  } else {
    digitalWrite(int1_ST, LOW);
    digitalWrite(int2_ST, HIGH);
    analogWrite(pwm_ST, speed);
  }
}
