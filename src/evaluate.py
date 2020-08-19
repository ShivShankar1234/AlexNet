from src.alexNet import AlexNet

"""To run:
alex_net = alexNet(params)
alex_net.run_experiment(N, large_data_set=False, generator=True
"""

def main():
    alex_net = AlexNet(input_width=227, input_height=227, input_channels=3, num_classes=10, learning_rate=0.01,
                       momentum=0.9, dropout_prob=0.5, weight_decay=0.0005)
    alex_net.run_experiment(n=10, large_data_set=False, generator=True)


if __name__ == '__main__':
    main()