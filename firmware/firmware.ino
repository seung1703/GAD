// ===================================================
//  통합 펌웨어 v5.0 — 주행 + 장애물 미션 공용 (단일 소스)
//
//  이 차는 아두이노가 하나이므로 펌웨어도 하나다. 미션별 파이썬 코드
//  (drive/ , obstacle/)는 분리돼 있지만 펌웨어는 이 파일 하나만 업로드한다.
//
//  [기존 v4.0 그대로]  조향/모터/워치독/1바이트 프로토콜.
//    비트7=1 조향: v=(b&0x7F) 0..127, 64=중립. 좌우 구간별 map으로 pot 목표 산출
//    비트7=0 모터: 0x00 정지, 0x01~0x3F 전진 b*4, 0x41~0x7F 후진 (b&0x3F)*4
//    워치독 400ms — 맥 serial_driver.py의 150ms 하트비트와 세트
//
//  [v5.0 추가]
//    · 초음파(전방 2개) 텔레메트리 — 기본 OFF. 맥이 켤 때만 측정/송신한다.
//        pulseIn 이 뻥 뚫린 트랙에서 15ms씩 루프를 막으므로, 주행 미션은
//        끈 채로 두어 루프를 깨끗하게 유지한다. 장애물 미션만 켠다.
//    · pot 텔레메트리("P:<값>", 10Hz) — 부하가 없어 항상 켜둠(조향 도달 확인용).
//
//  [제어 명령 — 0x40 이스케이프]
//    맥은 0x40("정지")을 실제로 절대 보내지 않는다(_drive_byte(0)=0x00, 후진=0x41~).
//    그래서 0x40 을 제어 이스케이프로 재활용한다: 0x40 다음 1바이트가 명령.
//      0x40 0x01 → 초음파 스트리밍 ON
//      0x40 0x00 → 초음파 스트리밍 OFF
//    (맥이 이 시퀀스를 안 보내면 초음파는 계속 꺼진 상태)
//
//  ★ US_TRIG / US_ECHO 핀은 실제 배선에 맞게 확인할 것 ★
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
const int STEER_LEFT    = 825;  // 좌측 최대
const int STEER_RIGHT   = 679;  // 우측 최대
const int STEER_NEUTRAL = 751;  // 이론상 정중앙

const int STEER_DEADBAND = 8;   // 오차 허용 범위 (좌우 대칭)
const int MIN_SPEED = 100;      // 조향 모터 최소 PWM
                                // (70은 무부하 한계선 — 주행 하중 실리면 풀락
                                //  근처 고저항 구간에서 스톨해 덜 꺾이는 원인)
const int MAX_SPEED = 255;      // 조향 모터 최대 PWM (스윙 속도 최대)
const float STEER_KP = 3.5;     // P 제어 비례 계수

// ==========================================
// 2.5 초음파 센서 — 전방 2개 (★ 실제 배선 핀으로 확인 ★)
// ==========================================
// 인덱스 0 = 전방 왼쪽, 1 = 전방 오른쪽 (맥 config.py us_front_ids=[0,1] 기준)
const uint8_t US_N = 2;
const uint8_t US_TRIG[US_N] = {23, 25};  // 0번 TRIG, 1번 TRIG
const uint8_t US_ECHO[US_N] = {22, 24};  // 0번 ECHO, 1번 ECHO
const uint32_t US_PERIOD_MS = 25;      // 25ms마다 센서 1개 측정 (순환, 센서당 20Hz)
const uint32_t US_TIMEOUT_US = 15000;  // 에코 대기 상한 (~2.5m)
uint32_t lastUsMs = 0;
uint8_t usIdx = 0;
bool usEnabled = false;   // ★ 기본 OFF. 맥의 0x40 0x01 명령으로만 켜짐 ★
uint32_t lastPotMs = 0;   // pot 값 텔레메트리("P:<값>") 주기 타이머

// ==========================================
// 3. 제어 변수
// ==========================================
const uint32_t WATCHDOG_MS = 400;
uint32_t lastRxMs = 0;
bool watchdogTripped = false;
bool expectCmd = false;   // 0x40 이스케이프: 다음 바이트가 제어 명령

int targetSteering = STEER_NEUTRAL;

void setup() {
  pinMode(int1_ST, OUTPUT); pinMode(int2_ST, OUTPUT); pinMode(pwm_ST, OUTPUT);
  pinMode(int1_LR, OUTPUT); pinMode(int2_LR, OUTPUT); pinMode(pwm_LR, OUTPUT);
  pinMode(int1_RR, OUTPUT); pinMode(int2_RR, OUTPUT); pinMode(pwm_RR, OUTPUT);
  for (uint8_t i = 0; i < US_N; i++) {
    pinMode(US_TRIG[i], OUTPUT);
    digitalWrite(US_TRIG[i], LOW);
    pinMode(US_ECHO[i], INPUT);
  }

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

  uint32_t now = millis();

  // 4. 초음파 라운드로빈 측정 + 송신 ("U<idx>:<cm>\n") — 켜져 있을 때만
  if (usEnabled && (now - lastUsMs >= US_PERIOD_MS)) {
    lastUsMs = now;
    int cm = usMeasure(usIdx);
    Serial.print('U');
    Serial.print(usIdx);
    Serial.print(':');
    Serial.println(cm);
    usIdx = (usIdx + 1) % US_N;
  }

  // 5. 조향 pot 텔레메트리 ("P:<값>", 100ms 주기) — 항상 켜둠 (부하 무시 가능)
  //    풀락 명령 시 실제로 어디까지 도달하는지 맥에서 확인용
  if (now - lastPotMs >= 100) {
    lastPotMs = now;
    Serial.print("P:");
    Serial.println(analogRead(potPin));
  }
}

// ===== 초음파 1회 측정 (cm, 실패=250) =====
int usMeasure(uint8_t i) {
  digitalWrite(US_TRIG[i], LOW);
  delayMicroseconds(2);
  digitalWrite(US_TRIG[i], HIGH);
  delayMicroseconds(10);
  digitalWrite(US_TRIG[i], LOW);
  unsigned long dur = pulseIn(US_ECHO[i], HIGH, US_TIMEOUT_US);
  if (dur == 0) return 250;            // 타임아웃(범위 밖/미연결)
  int cm = (int)(dur / 58);
  return cm < 2 ? 2 : (cm > 250 ? 250 : cm);
}

// ===== 1바이트 명령 처리 =====
void processByte(uint8_t b) {
  lastRxMs = millis();
  watchdogTripped = false;

  // 0x40 이스케이프: 직전 바이트가 0x40이면 이번 바이트는 제어 명령
  if (expectCmd) {
    expectCmd = false;
    usEnabled = (b != 0x00);   // 0x01=ON, 0x00=OFF
    return;
  }
  if (b == 0x40) {             // 이스케이프 프리픽스 (맥은 정지로는 0x00만 씀)
    expectCmd = true;
    return;
  }

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
    // ── 뒷바퀴 모터 ── (0x40은 위에서 이스케이프로 가로챔)
    if (b == 0x00) {
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

  // 우측(679) < 중립(751) < 좌측(825): error>0 = 왼쪽으로
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
