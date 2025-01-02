from TDoACalculation import TDoACalculation
import matplotlib.pyplot as plt
import numpy as np

class TDoAPlot:
    TrackerTracePointNum = 50
    
    def __init__(self):
        # Create instance of TDoACalculation class
        hyperbolic_parameter_max = 4.0
        self.tdoa_calculation = TDoACalculation(hyperbolic_parameter_max)

        # Create a figure with two subplots
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 12))

        # Plot with trajectories
        self.axis_trajectory = self.axs[0, 0]

        # Plot with 1D solution Histogram (X projection)
        self.axis_hist_x = self.axs[0, 1]
        # Plot with 1D solution Histogram (Y projection)
        self.axis_hist_y = self.axs[1, 0]
        # Plot with 2D solution Histogram
        self.axis_hist = self.axs[1, 1]

        for ax in [self.axis_trajectory, self.axis_hist]:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_ylim(-100, 100)
            ax.set_xlim(-100, 100)
            ax.grid(True, linestyle="--", alpha=0.7)

        for ax in [self.axis_hist_x, self.axis_hist_y]:
            ax.set_xlim(-100, 100)
            ax.grid(True, linestyle="--", alpha=0.7)

        self.axis_trajectory.title.set_text('Trackers Position')
        self.axis_hist.title.set_text('Time-Cumulative Hyperbolic Solution (XY)')
        self.axis_hist_x.title.set_text('Time-Cumulative Hyperbolic Solution (X-Axis)')
        self.axis_hist_y.title.set_text('Time-Cumulative Hyperbolic Solution (Y-Axis)')

        self.solution_line = self.axis_trajectory.plot([], [], label="Hyperbolic Solution", color="green")[0]
        self.plt_tracker_1 = self.axis_trajectory.scatter(0, 0, color="red", label=f"Tracker 1", s=20)
        self.plt_tracker_2 = self.axis_trajectory.scatter(0, 0, color="blue", label=f"Tracker 2", s=20)
            
        self.plt_tracker_1_trace = self.axis_trajectory.plot([], [], color="red", linestyle = '--')[0]
        self.plt_tracker_2_trace = self.axis_trajectory.plot([], [], color="blue", linestyle = '--')[0]
            
        self.plt_target = self.axis_trajectory.scatter([], [], color="green", label=f"Target", s=30)

        self.axis_trajectory.legend(loc='upper left')

        self.data_accum_x = list()
        self.data_accum_y = list()

        self.tracker_1_trace = list()
        self.tracker_2_trace = list()
    
    def add_solution(self, target: np.array, tracker_position_1: np.array, tracker_position_2: np.array):
        (x_inertial, y_inertial) = self.tdoa_calculation.find_hyperbolic_solution(target, tracker_position_1, tracker_position_2)

        self.data_accum_x.extend(x_inertial)
        self.data_accum_y.extend(y_inertial)

        self.tracker_1_trace.append([tracker_position_1[0], tracker_position_1[1]])
        self.tracker_2_trace.append([tracker_position_2[0], tracker_position_2[1]])

        if len(self.tracker_1_trace) > self.TrackerTracePointNum:
            self.tracker_1_trace.pop(0)
        if len(self.tracker_2_trace) > self.TrackerTracePointNum:
            self.tracker_2_trace.pop(0)

        self.solution_line.set_data(x_inertial, y_inertial)

    def calculate_histograms(self):
        # # Plot last position of trackers
        self.plt_tracker_1.set_offsets(self.tracker_1_trace[-1])
        self.plt_tracker_2.set_offsets(self.tracker_2_trace[-1])

        # Plot last trajectory trace of trackers
        self.plt_tracker_1_trace.set_data([elem[0] for elem in self.tracker_1_trace],
                                          [elem[1] for elem in self.tracker_1_trace])
        self.plt_tracker_2_trace.set_data([elem[0] for elem in self.tracker_2_trace],
                                          [elem[1] for elem in self.tracker_2_trace])

        data, x, y = np.histogram2d(self.data_accum_x, self.data_accum_y, range=[[-100, 100], [-100, 100]], bins = 200)
        hist_artist = self.axis_hist.imshow(data.T, interpolation = 'sinc', origin = 'lower', cmap='inferno', extent=[-100, 100, -100, 100])

        data, x, y = np.histogram2d(self.data_accum_x, self.data_accum_y, range=[[-100, 100], [-100, 100]], bins = 200)
        cumulative_sum_x = np.max(data, axis=0)
        cumulative_sum_y = np.max(data, axis=1)

        self.axis_hist_x.bar(x[:-1], cumulative_sum_x, width=1)
        self.axis_hist_y.bar(y[:-1], cumulative_sum_y, width=1)