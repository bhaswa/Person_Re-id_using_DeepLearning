require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

local lfs  = require 'lfs'
local t = require '../datasets/transforms'

local dataset_dir = '../../../../mnt/Datasets/i-LIDS-VID/sequences/'
local write_dir = '../../CNNFeatures34/'

local batch_size = 1

local cams = {'cam1', 'cam2'}
local skip = 1
for cam_num = 1, 2 do

  local cam_dir = dataset_dir .. cams[cam_num] .. '/'
  
  lfs.mkdir(write_dir .. cams[cam_num])
  for person in lfs.dir(cam_dir) do
    if person~="." and person~=".." then
--      print(person)
      local dir_path = cam_dir .. person
  
      lfs.mkdir(write_dir .. cams[cam_num] .. '/' .. person)
  
      for file in lfs.dir(dir_path) do
  
          if file~="." and file~=".." and file:sub(1,1)~='.' then
--	      print(file:sub(1,1))
              local list_of_filenames = {}
              table.insert(list_of_filenames, dir_path .. '/' .. file)
  
              local batch_size = #list_of_filenames
              local number_of_files = batch_size
  
              assert(batch_size == 1)
  
              -- Load the model
              local model = torch.load(arg[1]):cuda()
  
              -- Remove the fully connected layer
              assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
              model:remove(#model.modules)
  
              -- Evaluate mode
              model:evaluate()
  
              -- The model was trained with this input normalization
              local meanstd = {
                 mean = { 0.485, 0.456, 0.406 },
                 std = { 0.229, 0.224, 0.225 },
              }
  
              local transform = t.Compose{
                 t.Scale(256),
                 t.ColorNormalize(meanstd),
                 t.CenterCrop(224),
              }
  
              local features
  
              for i=1,number_of_files,batch_size do
                  local img_batch = torch.FloatTensor(batch_size, 3, 224, 224) -- batch numbers are the 3 channels and size of transform
  
                  -- preprocess the images for the batch
                  local image_count = 0
                  for j=1,batch_size do
                      img_name = list_of_filenames[i+j-1]
--		      print(img_name)  
                      if img_name  ~= nil then
                          image_count = image_count + 1
                          local img = image.load(img_name, 3, 'float')
                          img = transform(img)
                          img_batch[{j, {}, {}, {} }] = img
                      end
                  end
  
                  -- if this is last batch it may not be the same size, so check that
                  if image_count ~= batch_size then
                      img_batch = img_batch[{{1,image_count}, {}, {}, {} } ]
                  end
  
                 -- Get the output of the layer before the (removed) fully connected layer
                 local output = model:forward(img_batch:cuda()):squeeze(1)
  
  
                 -- this is necesary because the model outputs different dimension based on size of input
                 if output:nDimension() == 1 then output = torch.reshape(output, 1, output:size(1)) end
  
                 if not features then
                     features = torch.FloatTensor(number_of_files, output:size(2)):zero()
                 end
                     features[{ {i, i-1+image_count}, {}  } ]:copy(output)
  
              end
  
              torch.save(write_dir .. cams[cam_num] .. '/' .. person .. '/' .. file .. '_features.t7', {features=features, image_list=list_of_filenames})
          end
      end
      print('saved features for ' .. cam_num .. ' ' .. person)
    end
  end
end

