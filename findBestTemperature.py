from VGG_Transfer import main

temperatures = [.4,.6,1.0,1.5,2.0,5.0]

for t in temperatures:
    main([[],[]], [[],[]], 100, 1., 10., 0., t, .5e-4, 32, 10, False, "results_temp%f.txt"%(t))