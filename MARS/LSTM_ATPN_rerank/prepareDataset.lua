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

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'cunn'
require 'cutorch'
require 'paths'
require 'image'

prepareDataset = {}

--load all images into a flat list
local function loadSequenceImages(dirname,filesList)
	ImgSeq = {}
	local startIndex=1,count
	for c = 1, 6 do
		if filesList[c]~=nil then
			ImgSeq[c]={}
			for t = startIndex, startIndex+tablelength(filesList[c])-1 do
				if filesList[c][t]~=nil then
					ImgSeq[c][t]={}
					local nImgs = #filesList[c][t]
					for i,file in ipairs(filesList[c][t]) do
						local filename = paths.concat(dirname,file)
						local feat = torch.load(filename).features
						-- --allocate storage
						if i == 1 then
						    local s = #feat
						    featData = torch.DoubleTensor(nImgs, 1, 512)
						end

						feat = feat:float()
						local v = torch.sqrt(torch.var(feat[1]))
						local m = torch.mean(feat[1])
						feat[1] = feat[1] - m
						feat[1] = feat[1] / torch.sqrt(v)
						featData[{ {i}, {1}, {}}] = feat[1]
					end  
				end
				ImgSeq[c][t]=featData
			count=t
			end
		startIndex=count+1
		end
	end
    
    return ImgSeq
end

-- given a directory containing all images in a sequence get all the image filenames in order
local function getSequenceImageFiles(seqRoot,filesExt)

    local seqFiles = {}
    -- Go over all files in directory. We use an iterator, paths.files().
    for file in paths.files(seqRoot) do
       -- We only load files that match the extension
       if file:find(filesExt .. '$') and file:sub(1,1)~='.' then
          -- and insert the ones we care about in our table
          local cam = tonumber(file:sub(6,6))
          if seqFiles[cam]==nil then
          	seqFiles[cam]={}
          end
          local track = tonumber(file:sub(8,11))
          if seqFiles[cam][track]==nil then
          	seqFiles[cam][track]={}
          end
          table.insert(seqFiles[cam][track], file)
       end
    end
    
    -- Check files exist
    if tablelength(seqFiles) == 0 then
       error('given directory doesnt contain any files' .. seqRoot)
    end
    
    -- function used to sort the filenames
    local function numOrd(a,b)
        local in1 = 0
        in1 = tonumber(a:sub(13,15))

        local in2 = 0
        in2 = tonumber(b:sub(13,15))

        if in1 < in2 then
            return true
        else
            return false
        end
    end
    
    for _,t in ipairs(seqFiles) do
    	for _,img in ipairs(t) do
    		table.sort(img, numOrd)
    	end
    end
    
    return seqFiles
end

-- get a sorted list of directories for all the persons in the dataset
local function getPersonDirsList(seqRootDir)

    local tmpSeqCam = paths.concat(seqRootDir,'bbox_train')

    local personDirs = {}
    -- Go over all files in directory. We use an iterator, paths.files().
    for file in paths.files(tmpSeqCam) do
       -- We only load files that match the extension
       if #file > 2 and file:sub(1,1)~='.' then
          -- and insert the ones we care about in our table
           --print(file)
	   table.insert(personDirs, file)
       end
    end
    
    local tmpSeqCam = paths.concat(seqRootDir,'bbox_test')

    -- Go over all files in directory. We use an iterator, paths.files().
    for file in paths.files(tmpSeqCam) do
       -- We only load files that match the extension
       if #file > 2 and file:sub(1,1)~='.' then
          -- and insert the ones we care about in our table
           --print(file)
	   table.insert(personDirs, file)
       end
    end

    -- Check files exist
    if #personDirs == 0 then
       error(seqRootDir .. ' directory does not contain any image files')
    end
    table.sort(personDirs)

    return personDirs
end

-- return all images - we can later split this into the training / validation sets
function prepareDataset.prepareDataset(datasetRootDir,datasetRootDirOF,fileExt)
    local dataset = {}
    local personDirs = getPersonDirsList(datasetRootDir)
    for i,pdir in ipairs(personDirs) do
    	dataset[tonumber(pdir)] = {}
        local tracks = {}
        
        local seqRoot = paths.concat(paths.concat(datasetRootDir,'bbox_train'),pdir)
	if not paths.dirp(seqRoot) then
		seqRoot = paths.concat(paths.concat(datasetRootDir,'bbox_test'),pdir)
	end
	
        tracks = getSequenceImageFiles(seqRoot,fileExt)
        dataset[tonumber(pdir)] = loadSequenceImages(seqRoot,tracks)
    end
    return dataset
end

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

return prepareDataset
