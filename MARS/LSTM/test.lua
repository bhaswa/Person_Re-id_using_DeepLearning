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

-- standard method of computing the CMC curve using sequences

function computeCMC_MeanPool_LSTM(personImgs,cmcTestInds,net,outputSize,sampleSeqLength)

    net:evaluate()

    local nPersons = cmcTestInds:size(1)

    local avgSame = 0
    local avgDiff = 0
    local avgSameCount = 0
    local avgDiffCount = 0
	local cmc = torch.zeros(15,nPersons)
	local cnt = 1
	
	for cam1 = 1,5 do
		for cam2 = cam1+1, 6 do
			
			local cal = 1
			local feats_cam_a = {}
			local feats_cam_b = {}
			for i = 1,nPersons do
				if personImgs[cmcTestInds[i]][cam1]~=nil and personImgs[cmcTestInds[i]][cam2]~=nil then
					local tab = {}
					for _,tt in pairs(personImgs[cmcTestInds[i]][cam1]) do
						table.insert(tab,tt)
					end
					t = tab[math.random(1, #tab)]
					feats_cam_a[cal] = torch.DoubleTensor(outputSize)
					local actualSampleLen = 0
					local seqLen = t:size(1)
					if seqLen > sampleSeqLength then
						actualSampleLen = sampleSeqLength
					else
						actualSampleLen = seqLen
					end
					seq_length = actualSampleLen
					local seq = t[{{1,1 + (actualSampleLen - 1)},{},{}}]:squeeze():clone()
					local augSeq = {}
					for k = 1,actualSampleLen do
						local u = seq[{{k},{}}]:clone()
						u = u - torch.mean(u)
						augSeq[k] = u:cuda():clone()
					end
					feats_cam_a[cal] = net:forward(augSeq):double()
					
					tab = {}
					for _,tt in pairs(personImgs[cmcTestInds[i]][cam2]) do
						table.insert(tab,tt)
					end
					t = tab[math.random(1, #tab)]
					feats_cam_b[cal] = torch.DoubleTensor(outputSize)
					local actualSampleLen = 0
					local seqLen = t:size(1)
					if seqLen > sampleSeqLength then
						actualSampleLen = sampleSeqLength
					else
						actualSampleLen = seqLen
					end
					seq_length = actualSampleLen
					local seq = t[{{1,1 + (actualSampleLen - 1)},{},{}}]:squeeze():clone()
					local augSeq = {}
					for k = 1,actualSampleLen do
						local u = seq[{{k},{}}]:clone()
						u = u - torch.mean(u)
						augSeq[k] = u:cuda():clone()
					end
					feats_cam_b[cal] = net:forward(augSeq):double()
					cal = cal + 1
				end
			end
			
			cal = cal - 1
			
			local simMat = torch.zeros(cal,cal)
			local mAP = 0
			
			
			for i = 1,cal do
				local fa = feats_cam_a[i]
				for j = 1, cal do
					local fb = feats_cam_b[j]
					local dst = torch.sqrt(torch.sum(torch.pow(fa - fb,2)))
					simMat[i][j] = simMat[i][j] + dst
					if i == j then
						avgSame = avgSame  + dst
						avgSameCount = avgSameCount + 1
					else
						avgDiff = avgDiff + dst
						avgDiffCount = avgDiffCount + 1
					end
				end
			end

			avgSame = avgSame / avgSameCount
			avgDiff = avgDiff / avgDiffCount
			local cmcInds = torch.DoubleTensor(cal)
			local samplingOrder = torch.zeros(cal,cal)
    
			for i = 1,cal do
				--local AP = 0
				cmcInds[i] = i

				local tmp = simMat[{i,{}}]
				local y,o = torch.sort(tmp)

				--find the element we want
				local indx = 0
				local tmpIdx = 1
				local mapc = 0
				for j = 1,cal do
					if o[j] == i then
					    indx = j
					    break
					end
					    --mapc = mapc + 1
					    --AP = AP + (mapc/j)
					-- build the sampling order for the next epoch
					-- we want to sample close images i.e. ones confused with this person
					--else
					--    samplingOrder[i][tmpIdx] = o[j]
					--    tmpIdx = tmpIdx + 1
				end
			
				--if indx~=0 then 
					cmc[{{cnt},{indx,nPersons}}] = cmc[{{cnt},{indx,nPersons}}] + 1
				--end
			end
			cmc[cnt] = (cmc[cnt] / cal) * 100
			--mAP = mAP / cal
			cmcString = ''
			for c = 1,50 do
				if c <= nPersons then
				    cmcString = cmcString .. ' ' .. torch.floor(cmc[{cnt,c}])
				end
			end
			print(cmcString)
			--print(mAP)
			cnt = cnt + 1 
		end
	end
    return cmc/15,simMat,samplingOrder,avgSame,avgDiff
end
