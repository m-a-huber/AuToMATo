import paraview.simple as ps  # type: ignore

# create a new 'CSV Reader'
clusteringcsv = ps.CSVReader(FileName=["./eval/ttk_subclassed/tmp_in.csv"])

# create a new 'Table To Points'
tableToPoints1 = ps.TableToPoints(Input=clusteringcsv)
tableToPoints1.XColumn = "X"
tableToPoints1.YColumn = "Y"
tableToPoints1.a2DPoints = 1
tableToPoints1.KeepAllDataArrays = 1

# create a new 'Gaussian Resampling'
gaussianResampling1 = ps.GaussianResampling(Input=tableToPoints1)
gaussianResampling1.ResampleField = ["POINTS", "ignore arrays"]
gaussianResampling1.ResamplingGrid = [256, 256, 3]
gaussianResampling1.SplatAccumulationMode = "Sum"

# create a new 'Slice'
slice1 = ps.Slice(Input=gaussianResampling1)
slice1.SliceType = "Plane"

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Normal = [0.0, 0.0, 1.0]  # type: ignore

# create a new 'TTK PersistenceDiagram'
tTKPersistenceDiagram1 = ps.TTKPersistenceDiagram(Input=slice1)
tTKPersistenceDiagram1.ScalarField = ["POINTS", "SplatterValues"]
tTKPersistenceDiagram1.IgnoreBoundary = False

# save the output(s)
ps.SaveData("./eval/ttk_subclassed/tmp_out.csv", tTKPersistenceDiagram1)
