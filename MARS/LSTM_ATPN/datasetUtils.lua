-- Copyright (c) 2016 Niall McLaughlin, CSIT, Queen's University Belfast, UK
-- Contact: nmclaughlin02@qub.ac.uk
-- If you use this code please cite:
-- "Recurrent Convolutional Network for Video-based Person Re-Identification",
-- N McLaughlin, J Martinez Del Rincon, P Miller,
-- IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
--
-- This software is licensed for research and non-commercial use only.
-- 
-- The above copyright notice and this permission notice shall be included in
-- all copies or substantial portions of the Software.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
-- THE SOFTWARE.

local dataset_utils = {}

-- given the dataset, which consists of a table where t[x] contains the images for person x
-- split the dataset into testing and training parts
function dataset_utils.partitionDataset(seqRootDir)
    local tmpSeqCam = paths.concat(seqRootDir,'bbox_train')
	local trainDirs = {}
    -- Go over all files in directory. We use an iterator, paths.files().
    for file in paths.files(tmpSeqCam) do
       -- We only load files that match the extension
       if #file > 2 and file:sub(1,1)~='.' then
          -- and insert the ones we care about in our table
           --print(file)
	   table.insert(trainDirs, file)
       end
    end
    trainInds=torch.Tensor(trainDirs)
    
    local tmpSeqCam = paths.concat(seqRootDir,'bbox_test')
	local testDirs = {}
    -- Go over all files in directory. We use an iterator, paths.files().
    for file in paths.files(tmpSeqCam) do
       -- We only load files that match the extension
       if #file > 2 and file:sub(1,1)~='.' then
          -- and insert the ones we care about in our table
           --print(file)
	   table.insert(testDirs, file)
       end
    end
    testInds=torch.Tensor(testDirs)
    
    print('N train = ' .. trainInds:size(1))
    print('N test  = ' .. testInds:size(1))
    
    -- save the split to a file for later use
     datasetSplit = {
         trainInds = trainInds,
         testInds = testInds,
     }
     torch.save('./trainedNets/dataSplit.th7',datasetSplit)

    return trainInds,testInds
end

function gettrackdetails(dataset,trainInds,person,cam)
	local st = -1
	local co=0
	local T=dataset[trainInds[person]][cam]
	for i,_ in pairs(T) do
	if st==-1 then
		st=i
	end
	co = co + 1 end
	return st,co
end

-- the dataset format is dataset[person][camera][nSeq][nCrop][FeatureVec]
-- choose a pair of sequences from the same person
function dataset_utils.getPosSample(dataset,trainInds,person,sampleSeqLen)

    -- choose the camera, ilids video only has two, but change this for other datasets
    ::repeat1::
    local camA = torch.random(1,6)
    if dataset[trainInds[person]][camA]==nil then
    	goto repeat1
    end
    
    ::repeat2::
	local camB = torch.random(1,6)
	if dataset[trainInds[person]][camB]==nil then
    	goto repeat2
    end
    
    local trackA,trackB
    local starttrackA,counttrackA,starttrackB,counttrackB
    starttrackA,counttrackA=gettrackdetails(dataset,trainInds,person,camA)
    starttrackB,counttrackB=gettrackdetails(dataset,trainInds,person,camB)
    
    ::repeatA::
    trackA=torch.random(starttrackA,starttrackA+counttrackA-1)
    if dataset[trainInds[person]][camA][trackA]==nil then
    	goto repeatA
    end
    
    ::repeatB::
    trackB=torch.random(starttrackB,starttrackB+counttrackB-1)
    if dataset[trainInds[person]][camB][trackB]==nil then
    	goto repeatB
    end
    
    local actualSampleSeqLen = sampleSeqLen
    local nSeqA = dataset[trainInds[person]][camA][trackA]:size(1)
    local nSeqB = dataset[trainInds[person]][camB][trackB]:size(1)
    -- what to do if the sequence is shorter than the sampleSeqLen
    if nSeqA <= sampleSeqLen or nSeqB <= sampleSeqLen then
        if nSeqA < nSeqB then
            actualSampleSeqLen = nSeqA
        else
            actualSampleSeqLen = nSeqB
        end
    end

    local startA = torch.floor(torch.rand(1)[1] * ((nSeqA - actualSampleSeqLen) + 1)) + 1
    local startB = torch.floor(torch.rand(1)[1] * ((nSeqB - actualSampleSeqLen) + 1)) + 1

    return startA,startB,camA,camB,trackA,trackB,actualSampleSeqLen
end

-- the dataset format is dataset[person][camera][nSeq][nCrop][FeatureVec]
-- choose a pair of sequences from different people
function dataset_utils.getNegSample(dataset,trainInds,sampleSeqLen)

    local permAllPersons = torch.randperm(trainInds:size(1))
    local personA = permAllPersons[1]--torch.floor(torch.rand(1)[1] * 2) + 1
    local personB = permAllPersons[2]--torch.floor(torch.rand(1)[1] * 2) + 1
    -- choose the camera, ilids video only has two, but change this for other datasets
    ::repeat1::
    local camA = torch.random(1,6)
    if dataset[trainInds[personA]][camA]==nil then
    	goto repeat1
    end
    ::repeat2::
	local camB = torch.random(1,6)
	if dataset[trainInds[personB]][camB]==nil then
    	goto repeat2
    end
    local trackA,trackB
    local starttrackA,counttrackA,starttrackB,counttrackB
    starttrackA,counttrackA=gettrackdetails(dataset,trainInds,personA,camA)
    starttrackB,counttrackB=gettrackdetails(dataset,trainInds,personB,camB)
    ::repeatA::	
    trackA=torch.random(starttrackA,starttrackA+counttrackA-1)
    if dataset[trainInds[personA]][camA][trackA]==nil then
    	goto repeatA
    end
    ::repeatB::
    trackB=torch.random(starttrackB,starttrackB+counttrackB-1)
    if dataset[trainInds[personB]][camB][trackB]==nil then
    	goto repeatB
    end
    
    local actualSampleSeqLen = sampleSeqLen
    local nSeqA = dataset[trainInds[personA]][camA][trackA]:size(1)
    local nSeqB = dataset[trainInds[personB]][camB][trackB]:size(1)

    -- what to do if the sequence is shorter than the sampleSeqLen
    if nSeqA <= sampleSeqLen or nSeqB <= sampleSeqLen then
        if nSeqA < nSeqB then
            actualSampleSeqLen = nSeqA
        else
            actualSampleSeqLen = nSeqB
        end
    end

    local startA = torch.floor(torch.rand(1)[1] * ((nSeqA - actualSampleSeqLen) + 1)) + 1
    local startB = torch.floor(torch.rand(1)[1] * ((nSeqB - actualSampleSeqLen) + 1)) + 1

    return personA,personB,camA,camB,trackA,trackB,startA,startB,actualSampleSeqLen
end

return dataset_utils
