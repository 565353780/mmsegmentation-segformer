#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import os
import time

class MMSegmentationDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.valid_dataset_name_list = ['ade', 'city']
        self.dataset_name = None
        self.config = None
        self.checkpoint = None
        self.checkpoint_ready = False
        self.device = None
        self.model = None
        self.model_ready = False
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0

    def resetTimer(self):
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0

    def startTimer(self):
        self.time_start = time.time()

    def endTimer(self, save_time=True):
        time_end = time.time()

        if not save_time:
            return

        if self.time_start is None:
            print("startTimer must run first!")
            return

        if time_end > self.time_start:
            self.total_time_sum += time_end - self.time_start
            self.detected_num += 1
        else:
            print("Time end must > time start!")

    def getAverageTimeMS(self):
        if self.detected_num == 0:
            return -1

        return int(1000.0 * self.total_time_sum / self.detected_num)

    def getAverageFPS(self):
        if self.detected_num == 0:
            return -1

        return int(1.0 * self.detected_num / self.total_time_sum)

    def setCheckPoint(self, checkpoint):
        self.checkpoint = checkpoint
        self.checkpoint_ready = False

        checkpoint_file_name = os.path.basename(checkpoint)

        dataset_index = -1

        for i in range(len(self.valid_dataset_name_list)):
            if self.valid_dataset_name_list[i] in checkpoint_file_name:
                dataset_index = i
                break
         
        if dataset_index == -1:
            print("Cannot load this checkpoint type!")
            return

        self.dataset_name = self.valid_dataset_name_list[dataset_index]

        self.checkpoint_ready = True
        return

    def initEnv(self, config, checkpoint, device):
        self.reset()
        self.config = config
        self.device = device
        self.setCheckPoint(checkpoint)
        return


    def loadModel(self, config, checkpoint, device="cuda:0"):
        self.initEnv(config, checkpoint, device)

        if not self.checkpoint_ready:
            return

        self.model = init_segmentor(self.config, self.checkpoint, self.device)

        self.model_ready = True

        return

    def detect(self, image):
        result = inference_segmentor(self.model, image)
        return result

    def test(self, image_folder_path, run_episode=-1, timer_skip_num=5):
        if not self.model_ready:
            print("Model not ready yet, Please loadModel or check your model path first!")
            return

        if run_episode == 0:
            print("No detect run with run_episode=0!")
            return

        file_name_list = os.listdir(image_folder_path)
        image_file_name_list = []
        for file_name in file_name_list:
            if file_name[-4:] in [".jpg", ".png"]:
                image_file_name_list.append(file_name)

        if run_episode < 0:
            self.resetTimer()
            timer_skipped_num = 0

            while True:
                for image_file_name in image_file_name_list:
                    image_file_path = os.path.join(image_folder_path, image_file_name)

                    self.startTimer()

                    result = self.detect(image_file_path)

                    if timer_skipped_num < timer_skip_num:
                        self.endTimer(False)
                        timer_skipped_num += 1
                    else:
                        self.endTimer()

                    print("\rNet: SegFormer" +
                          "\tDetected: " + str(self.detected_num) +
                          "\tAvgTime: " + str(self.getAverageTimeMS()) + "ms"
                          "\tAvgFPS: " + str(self.getAverageFPS()) +
                          "    ", end="")

                    #  show_result_pyplot(self.model, image_file_path, result, get_palette(self.dataset_name))

            print()

            return

        self.resetTimer()
        total_num = run_episode * len(image_file_name_list)
        timer_skipped_num = 0

        for i in range(run_episode):
            for image_file_name in image_file_name_list:
                image_file_path = os.path.join(image_folder_path, image_file_name)

                self.startTimer()

                result = self.detect(image_file_path)

                if timer_skipped_num < timer_skip_num:
                    self.endTimer(False)
                    timer_skipped_num += 1
                else:
                    self.endTimer()

                print("\rNet: SegFormer" +
                      "\tDetected: " + str(self.detected_num) + "/" + str(total_num - timer_skip_num) +
                      "\t\tAvgTime: " + str(self.getAverageTimeMS()) + "ms"
                      "\tAvgFPS: " + str(self.getAverageFPS()) +
                      "    ", end="")

                #  show_result_pyplot(self.model, image_file_path, result, get_palette(self.dataset_name))

        print()


if __name__ == '__main__':
    config = "../SegFormer/local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py"
    checkpoint = "./segformer.b5.640x640.ade.160k.pth"
    image_folder_path = "./sample_images/"

    mm_segmentation_detector = MMSegmentationDetector()

    mm_segmentation_detector.loadModel(config, checkpoint)

    mm_segmentation_detector.test(image_folder_path)

