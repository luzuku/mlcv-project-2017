from VGG_Transfer import main


small_features_intermediate = [0, 3, 6, 9, 12]
big_features_intermediate = [0, 5, 10, 17, 24]

#t = 2.5
#main([[6],[]], [[10],[]], 40, 1., 10., 10., t, .2e-4, 25, 10, False, "experiment1.txt", "SGD")
#main([[9],[]], [[17],[]], 40, 1., 10., 10., t, .2e-4, 25, 10, False, "experiment2.txt", "SGD")
#main([[3],[]], [[5],[]], 40, 1., 10., 10., t, .2e-4, 25, 10, False, "experiment3.txt", "SGD")
#main([[],[0]], [[],[0]], 40, 1., 10., 10., t, .2e-4, 25, 10, False, "experiment4.txt", "SGD")
#main([[],[0]], [[],[3]], 40, 1., 10., 10., t, .2e-4, 25, 10, False, "experiment5.txt", "SGD")

t = 2.0
main([[9], []], [[17], []], 40, 1., 10., 10., t, .2e-4, 25, 10, False, "experiment6.txt", "SGD")
main([[3, 6, 9, 12], []], [[5, 10, 17, 24], []], 40, 1., 10., 10., t, .2e-4, 25, 10, False, "experiment7.txt", "SGD")
main([[3, 6, 9, 12], []], [[5, 10, 17, 24], []], 40, 1., 10., 40., t, .2e-4, 25, 10, False, "experiment8.txt", "SGD")
main([[12], []], [[24], []], 40, 1., 10., 10., t, .2e-4, 25, 10, False, "experiment9.txt", "SGD")
main([[6, 9], []], [[10, 17], []], 40, 1., 10., 10., t, .2e-4, 25, 10, False, "experiment10.txt", "SGD")
main([[0], []], [[0], []], 40, 1., 10., 10., t, .2e-4, 25, 10, False, "experiment11.txt", "SGD")