import cv2
import time
from ultralytics import YOLO

def benchmark(model_path, video_path, warmup=8, runs=30):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    model_name = model_path.replace('.', '_')
    out_path = f"output_{model_name}.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,(width, height))

    ret,frame_first = cap.read()
    
    if not ret:
        print("读取失败")
        return 0


    #预热
    print(f"预热{warmup}次")
    for _ in range(warmup):
        _ = model(frame_first, verbose=False,device='cuda:0')


    #测速
    print("正在测速...")
    total_time = 0
    for _ in range(runs):
        start = time.time()
        model(frame_first, verbose=False, device='cuda:0')
        total_time+=(time.time()-start)
    avg = (total_time/runs)*1000
    print(f"平均耗时: {avg:.2f} ms")


    #推理
    print("开始处理完整视频并保存结果")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_first_saved=False
    img_name = ""

    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False, device='cuda:0')
        annotated_frame = results[0].plot() 

        out.write(annotated_frame)
        if not frame_first_saved:
            img_name = f"screenshot_{model_path.split('.')[-1]}.jpg"
            cv2.imwrite(img_name, annotated_frame)
            frame_first_saved = True
            
    cap.release()
    out.release()

    print(f"视频结果已保存至：{out_path}")
    print(f"结果截图已保存至: {img_name}")
          
    return avg


#分别对比
if __name__ == "__main__":
    video = "raw_video.mp4"
    pt_avg = benchmark('best.pt', video)
    onnx_avg = benchmark('best.onnx', video)
    engine_avg = benchmark('best.engine', video)

    print("\n===总结===")
    print(f"PT 耗时: {pt_avg:.2f} ms/帧")
    print(f"ONNX 耗时: {onnx_avg:.2f} ms/帧")
    print(f"TensorRT Engine 耗时: {engine_avg:.2f} ms/帧")