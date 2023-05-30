from package import *

def print_fps(start_time):
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    fps_str = "FPS : %0.2f" %fps
    print(fps_str)

def predict(
        imgs=None,
        model=None,
        conf=0.5,
) -> list:
    """
        return:
        True -> 유효표
        False -> 무효표

        names: 
        0. 1 -> 1번 구역
        1. 2 -> 2번 구역
        2. 3 -> 3번 구역
        3. 4 -> 4번 구역
        4. stamp -> 정규 도장
        5. sign -> 도장이 아닌 글자 또는 낙서
        6. fake -> 정규 도장이 아닌 도장
        7. finger -> 지장

        role:
        1. stamp가 존재하지 않으면 무효표
        2. sign, fake가 존재한다면 무효표
        3. stamp가 하나의 여러 구역에 존재하면 무효표
        4. finger는 있어도 없어도 상관 없음

    """
    start_time = time.time()
    outputs = "n"
    results = model.predict(source=imgs, verbose=False, iou=0.9, device=0)  # predict on an image
    stamps, areas = [], {}

    for result in results:
        score = result.boxes.conf.cpu().numpy()  # 객체 당 score tensor 리스트
        classes = result.boxes.cls.cpu().numpy()  # 객체 당 class 리스트
        xyxy = result.boxes.xyxy.cpu().numpy()  # 객체 당 xyxy 리스트

        for i in range(len(classes)):

            if classes[i] == 5 or classes[i] == 6:
                print_fps(start_time)
                print("exist sign or fake")
                print("output: n")
                return "n"
            
            x1, y1, x2, y2 = xyxy[i]
            if (classes[i] == 0 or classes[i] == 1 or classes[i] == 2 or classes[i] == 3) and score[i] > conf:
                areas[int(classes[i])] = box(x1, y1, x2, y2)
            elif classes[i] == 4 and score[i] > conf:
                stamps.append(box(x1, y1, x2, y2))

    stamp_val = []

    for stamp in stamps:
        dup_val = 0
        for key, area in areas.items():
            intersection = (stamp & area).area
            if intersection > 0.0:
                if key == 0:
                    # print(1)
                    dup_val += 1
                    outputs = "1"
                elif key == 1:
                    # print(2)
                    dup_val += 1
                    outputs = "2"
                elif key == 2:
                    # print(3)
                    dup_val += 1
                    outputs = "3"
                elif key == 3:
                    # print(4)
                    dup_val += 1
                    outputs = "4"
        if dup_val > 1:
            print_fps(start_time)
            print("dup")
            print("output: n")
            return "n"
        stamp_val.append(outputs)
        
    if len(set(stamp_val)-set("n")) != 1:
        print_fps(start_time)
        print("stamp val")
        print("output: n")
        return "n"

    print_fps(start_time)
    print(f"output: {outputs}")
    return outputs

if __name__ == "__main__":
    model_path = "./weights/medium.pt"

    model = YOLO(model=model_path, task="detect")

    total = defaultdict(int)

    ##### streaming #####
    start_cnt = time.time()
    cap = cv2.VideoCapture(0)
    while(True):
        cnt = int(time.time()-start_cnt) + 1
        ret, frame = cap.read() 
        cnt_position = (10,50)
        text_position = (50,50)
        text = json.dumps(dict(total))
        cv2.putText(frame, str(cnt), cnt_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        cv2.imshow('Push Enter key, Exit q key', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
        if cnt == 5:
            outputs = predict(imgs=frame, model=model, conf=0.5)
            if outputs:
                total[outputs] += 1
                print(total)
                start_cnt = time.time()
            continue
    cap.release()
    cv2.destroyAllWindows()

    ##### test #####

    # for img in glob.glob("./datasets/vote/test/*/*"):
    #     print('='*50)
    #     print(img)
    #     outputs = detect_pytorch(imgs=img, model=model, conf=0.7)
    #     print('='*50)
