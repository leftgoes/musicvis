import cv2
import numpy as np
import time
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from audio2numpy import open_audio
import matplotlib.pyplot as plt

from typing import Callable


ParameterCallable = Callable[[float], float]
ColorTuple = tuple[int, int, int]

class AttackRelease:
    def __init__(self, attack: int, release: int, attack_power: float, release_power: float) -> None:
        self.attack = abs(attack)
        self.release = abs(release)
        self.attack_power = abs(attack_power)
        self.release_power = abs(release_power)
    
    @property
    def _kernel_x(self) -> np.ndarray:
        return np.arange(0, self.attack + self.release + 1, dtype=np.float_)

    def _weight_function(self, x: np.ndarray) -> np.ndarray:
        return np.piecewise(x,
                            condlist=[x <= self.attack, self.attack < x],
                            funclist=[lambda t: (t / self.attack) ** self.attack_power, lambda t: (-(t - self.attack - self.release) / self.release) ** self.release_power])

    def kernel(self) -> np.ndarray:
        unnormlalized = self._weight_function(self._kernel_x)
        return unnormlalized / unnormlalized.sum()
    
    def apply_mono(self, audio: np.ndarray, delta: int = -1) -> np.ndarray:
        if delta == -1:
            delta = np.round((np.dot(self._kernel_x, self.kernel()) + self.attack) / 2).astype(int)
        return np.convolve(audio, self.kernel(), mode='full')[delta:delta + audio.shape[0]]

    def apply_stereo(self, audio: np.ndarray, delta: int = -1) -> np.ndarray:
        audio[:, 0] = self.apply_mono(audio[:, 0], delta)
        audio[:, 1] = self.apply_mono(audio[:, 1], delta)
        return audio

    def show_plot(self) -> None:
        plt.plot(self.kernel())
        plt.show()


class Video:
    def __init__(self, width: int, height: int, fps: float) -> None:
        self.width = width
        self.height = height
        self.fps = fps

        self.sample_rate: int = None

    def save(self, videopath: str, 
                   audiopath: str,
                   fg_imgpath: str, bg_imgpath: str | None = None,
                   mainaudio_start_relative: float = 0, mainaudio_end_relative: float = 1,
                   main_ar: AttackRelease | None = None,
                   parameter_audiopath: str | None = None,
                   parameter_ar: AttackRelease | None = None, parameter_timeblur_sigma: float = 0,
                   parameter_power: float = 1,
                   bg_blur: ParameterCallable | None = None,
                   bg_brightness: ParameterCallable | None = None,
                   bg_scale: ParameterCallable | None = None,
                   bar_width_px: int = 50,
                   bar_alpha: ParameterCallable | None = None,
                   bar_x_relative: float = 0.2) -> None:
        if not bg_imgpath: bg_imgpath = fg_imgpath
        if not main_ar: main_ar = AttackRelease(2, 40, 3, 3)
        if not parameter_ar: parameter_ar = AttackRelease(2, 20, 3, 3)
        if not bg_blur: bg_blur = lambda t: round(55 - 5 * t)
        if not bg_brightness: bg_brightness = lambda t: 0.3 + 0.1 * t
        if not bg_scale: bg_scale = lambda t: 1 + 0.2 * t
        if not bar_alpha: bar_alpha = lambda lr: 0.5 + 0.2 * lr

        # load foreground and background images
        foreground = cv2.imread(fg_imgpath)
        foreground = cv2.resize(foreground,
                                dsize=(round(0.83 * self.height * foreground.shape[1] / foreground.shape[0]), round(0.83 * self.height)),
                                interpolation=cv2.INTER_CUBIC)
        
        background = cv2.imread(bg_imgpath)

        # load main audio
        _mainaudio_raw, self.sample_rate = open_audio(audiopath)
        _mainaudio_start_index, _mainaudio_end_index = int(mainaudio_start_relative * _mainaudio_raw.shape[0]), int(mainaudio_end_relative * _mainaudio_raw.shape[0])
        mainaudio: np.ndarray = _mainaudio_raw[_mainaudio_start_index:_mainaudio_end_index]  # cut

        samples_count = mainaudio.shape[0]
        frames_count = int(np.ceil(self.fps * mainaudio.shape[0] / self.sample_rate))
        frame_start = lambda n: int(n * self.sample_rate / self.fps)

        # load background scaler audio
        _paramaudio_raw, param_sample_rate = open_audio(parameter_audiopath)
        assert param_sample_rate == self.sample_rate
        _paramaudio_raw: np.ndarray = _paramaudio_raw.mean(axis=1) ** 2  # stereo to mono and get energy

        # cut and make sure main and background scaler audio have same dims
        paramaudio = np.zeros(mainaudio.shape[0:1])
        if _mainaudio_end_index > _paramaudio_raw.shape[0]:
            paramaudio[:samples_count - (_mainaudio_end_index - _paramaudio_raw.shape[0])] = _paramaudio_raw[_mainaudio_start_index:]
        else:
            paramaudio = _paramaudio_raw[_mainaudio_start_index:_mainaudio_end_index]

        # calculate stereo main audio energy
        main_energys = np.empty((frames_count, 2))
        for i in range(frames_count):
            if len(mainaudio.shape) == 2:  # main audio is stereo
                main_energys[i] = 2 * (mainaudio[frame_start(i):frame_start(i + 1)] ** 2).mean(axis=0)
            else:
                main_energys[i, :] = 2 * (mainaudio[frame_start(i):frame_start(i + 1)] ** 2).mean()
        main_energys = main_ar.apply_stereo(main_energys)
        main_energys /= main_energys.max()

        # calculate parameter series
        param_energys = np.zeros(frames_count)
        for i in range(frames_count):
            param_energys[i] = paramaudio[frame_start(i) : frame_start(i + 1)].mean()
        
        if parameter_timeblur_sigma != 0:
            param_energys = gaussian_filter1d(param_energys, parameter_timeblur_sigma)
        param_energys = parameter_ar.apply_mono(param_energys)  # add (temporal) tail to scale factores and apply power
        param_energys = (param_energys / param_energys.max()) ** parameter_power  # normalize
        plt.plot(param_energys)
        plt.show()
        
        _start = time.perf_counter()
        # write to video file
        videoout = cv2.VideoWriter(videopath, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height), True)
        for i, (t, (l, r)) in enumerate(zip(param_energys, main_energys)):
            _progress = (i + 1) / frames_count
            _min, _sec = divmod((1/_progress - 1) * (time.perf_counter() - _start), 60)
            print(f'\rProcessing frame {i + 1}/{frames_count} ({round(100 * _progress)}%), remaining est. {_min}m{round(_sec)}s...', end='')

            frame = cv2.resize(background,
                               dsize=(round(bg_scale(t) * self.width), round(bg_scale(t) * self.width * background.shape[0] / background.shape[1])),
                               interpolation=cv2.INTER_CUBIC)
            
            # fit background to video dims
            bg_left, bg_lower = (frame.shape[1] - self.width) // 2, (frame.shape[0] - self.height) // 2
            frame = frame[bg_lower:bg_lower + self.height, bg_left:bg_left + self.width]

            # apply fx to background
            frame = np.uint8(bg_brightness(t) * cv2.blur(frame, ksize=(round(bg_blur(t)), round(bg_blur(t)))))

            # draw foreground
            fg_height, fg_width, _ = foreground.shape
            fg_left, fg_lower = (self.width - fg_width) // 2, (self.height - fg_height) // 2
            frame[fg_lower:fg_lower + fg_height, fg_left:fg_left + fg_width] = foreground

            # draw levels
            l_bar_height, r_bar_height = round(l * fg_height), round(r * fg_height)
            l_bar_xcenter, r_bar_xcenter = round((1 - bar_x_relative) * fg_left), round((1 + bar_x_relative) * fg_left + fg_width)

            frame[-(fg_lower + l_bar_height):-fg_lower, l_bar_xcenter - bar_width_px // 2:l_bar_xcenter + bar_width_px // 2] = bar_alpha(l) * 255 + (1 - bar_alpha(l)) * frame[-(fg_lower + l_bar_height):-fg_lower, l_bar_xcenter - bar_width_px // 2:l_bar_xcenter + bar_width_px // 2]  # draw left bar
            frame[-(fg_lower + r_bar_height):-fg_lower, r_bar_xcenter - bar_width_px // 2:r_bar_xcenter + bar_width_px // 2] = bar_alpha(r) * 255 + (1 - bar_alpha(r)) * frame[-(fg_lower + r_bar_height):-fg_lower, r_bar_xcenter - bar_width_px // 2:r_bar_xcenter + bar_width_px // 2]  # draw right bar

            # write frame
            videoout.write(frame)
        videoout.release()
        print(f'\nfinished in {time.perf_counter() - _start:.02f} seconds')