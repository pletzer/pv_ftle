import vtk

# ----------------------------------------------------------------------
# Input file
# ----------------------------------------------------------------------
filename = "wrf_30.nc" #"/Users/apletzer/work/pv_ftle/wrfout_d03_2013010900_f030.nc"

# ----------------------------------------------------------------------
# Create NetCDF CF reader (WRF-compatible)
# ----------------------------------------------------------------------
reader = vtk.vtkNetCDFCFReader()
reader.SetFileName(filename)
reader.Update()

output = reader.GetOutput()

print("====================================================")
print("WRF NetCDF file:", filename)
print("====================================================")

# ----------------------------------------------------------------------
# Handle multiblock datasets (WRF usually is)
# ----------------------------------------------------------------------
if isinstance(output, vtk.vtkMultiBlockDataSet):
    print("Dataset type: vtkMultiBlockDataSet")
    print("Number of blocks:", output.GetNumberOfBlocks())
    print()

    for i in range(output.GetNumberOfBlocks()):
        block = output.GetBlock(i)
        name = output.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())

        print(f"--- Block {i} ---")
        print("Name:", name)
        print("Type:", block.GetClassName())

        if isinstance(block, vtk.vtkDataSet):
            print("Dimensions:", block.GetDimensions() if hasattr(block, "GetDimensions") else "N/A")
            print("Bounds:", block.GetBounds())

            # ----------------------------------------------------------
            # Point data arrays
            # ----------------------------------------------------------
            pd = block.GetPointData()
            print("Point data arrays:")
            for j in range(pd.GetNumberOfArrays()):
                arr = pd.GetArray(j)
                print(f"  - {arr.GetName()} ({arr.GetNumberOfComponents()} components)")

            # ----------------------------------------------------------
            # Cell data arrays
            # ----------------------------------------------------------
            cd = block.GetCellData()
            print("Cell data arrays:")
            for j in range(cd.GetNumberOfArrays()):
                arr = cd.GetArray(j)
                print(f"  - {arr.GetName()} ({arr.GetNumberOfComponents()} components)")

        print()

else:
    # ------------------------------------------------------------------
    # Single dataset case
    # ------------------------------------------------------------------
    print("Dataset type:", output.GetClassName())
    print("Bounds:", output.GetBounds())

    print("\nPoint data arrays:")
    pd = output.GetPointData()
    for i in range(pd.GetNumberOfArrays()):
        arr = pd.GetArray(i)
