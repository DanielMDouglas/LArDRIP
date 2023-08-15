import h5py

class MAEdataloader:
    def __init__(self, patched_image_input, maskFraction = 0.5, batchSize = 16):
        self.patched_image_input = patched_image_input

        self.maskFraction = maskFraction

    def masker(self, patches):
        # take a list of jumbled patches and select a fraction
        # of them
        # return the unmasked patches and their coordinate key
        # and return the coordinate keys of the masked patches

        # return unmaskedPatches, unmaskedKeys, maskedKeys

        nPatches = len(patches)
        nKept = int((1 - self.maskFraction)*nPatches)

        patchChoice = np.random.choice(len(patches),
                                       size = nKept,
                                       replace = False)

        keptPatches = []
        maskedPatches = []
        for thisPatch in patches:
            if thisPatch[0] in patchChoice:
                keptPatches.append(thisPatch)
            else:
                maskedPatches.append(thisPatch)
        
        # print ("patch choice", patchChoice)
        # print (len(patches))
        return keptPatches, maskedPatches
        
    def load_image(self): 
        return next_image

    def __getitem__(self, idx):
        
    
    def __iter__(self):
        return
