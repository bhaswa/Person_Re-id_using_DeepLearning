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

function buildModel_MeanPool_LSTM(nFltrs1,nFltrs2,nFltrs3,nPersonsTrain)

    local nFilters = {nFltrs1,nFltrs2,nFltrs3}

    local filtsize = {5,5,5}
    local poolsize = {2,2,2}
    local stepSize = {2,2,2}


    -- remember this adds padding to ALL SIDES of the image
    local padDim = 4

    local cnn = nn.Sequential()

    local numhidden = 256

    -- local ninputChannels = 3
    -- cnn:add(nn.SpatialZeroPadding(padDim, padDim, padDim, padDim))
    -- cnn:add(nn.SpatialConvolutionMM(ninputChannels, nFilters[1], filtsize[1], filtsize[1], 1, 1))
    -- cnn:add(nn.Tanh())
    -- cnn:add(nn.SpatialMaxPooling(poolsize[1],poolsize[1],stepSize[1],stepSize[1]))
    --
    -- ninputChannels = nFilters[1]
    -- cnn:add(nn.SpatialZeroPadding(padDim, padDim, padDim, padDim))
    -- cnn:add(nn.SpatialConvolutionMM(ninputChannels, nFilters[2], filtsize[2], filtsize[2], 1, 1))
    -- cnn:add(nn.Tanh())
    -- cnn:add(nn.SpatialMaxPooling(poolsize[2],poolsize[2],stepSize[2],stepSize[2]))
    --
    -- ninputChannels = nFilters[2]
    -- cnn:add(nn.SpatialZeroPadding(padDim, padDim, padDim, padDim))
    -- cnn:add(nn.SpatialConvolutionMM(ninputChannels, nFilters[3], filtsize[3], filtsize[3], 1, 1))
    -- cnn:add(nn.Tanh())
    -- cnn:add(nn.SpatialMaxPooling(poolsize[3],poolsize[3],stepSize[3],stepSize[3]))

    local nFullyConnected = 512

--    cnn:add(nn.Reshape(1,nFullyConnected))
    cnn:add(nn.Dropout(opt.dropoutFrac))
    cnn:add(nn.Linear(nFullyConnected,opt.embeddingSize))
    --cnn:cuda()

    -- local h2h = nn.Sequential()
    -- h2h:add(nn.Tanh())
    -- h2h:add(nn.Dropout(opt.dropoutFracRNN))
    -- h2h:add(nn.Linear(opt.embeddingSize,opt.embeddingSize))
    -- h2h:cuda()
    --
    -- local r1 = nn.Recurrent(
    --     opt.embeddingSize,
    --     cnn,
    --     h2h,
    --     nn.Identity(),
    --     opt.sampleSeqLength)
    lstm1 = nn.Sequential()
    lstm1:add(cnn)

    lstm1:add(nn.FastLSTM(opt.embeddingSize, opt.embeddingSize, numHidden))
    -- lstm1:add(nn.View(-1, numHidden))
    if opt.dropoutLSTM > 0 then
      lstm1:add(nn.Dropout(opt.dropoutLSTM))
    end

    local lstm_p1 = nn.Sequencer(
        nn.Sequential()
        :add(lstm1)
        )

    Combined_CNN_LSTM_1 = nn.Sequential()
    Combined_CNN_LSTM_1:add(lstm_p1)
    Combined_CNN_LSTM_1:add(nn.JoinTable(1))
    --Combined_CNN_LSTM_1:add(nn.Mean(1))

    -- local r2 = nn.Recurrent(
    --     opt.embeddingSize,
    --     cnn:clone('weight','bias','gradWeight','gradBias'),
    --     h2h:clone('weight','bias','gradWeight','gradBias'),
    --     nn.Identity(),
    --     opt.sampleSeqLength)

    lstm2 = nn.Sequential()
    lstm2:add(cnn:clone('weight','bias','gradWeight','gradBias'))
    lstm2:add(nn.FastLSTM(opt.embeddingSize, opt.embeddingSize, numHidden))
    -- lstm2:add(nn.View(-1, numHidden))
    if opt.dropoutLSTM > 0 then
      lstm2:add(nn.Dropout(opt.dropoutLSTM))
    end

    local lstm_p2 = nn.Sequencer(
        nn.Sequential()
        :add(lstm2)
        )

    Combined_CNN_LSTM_2 = nn.Sequential()
    Combined_CNN_LSTM_2:add(lstm_p2)
    Combined_CNN_LSTM_2:add(nn.JoinTable(1))
    --Combined_CNN_LSTM_2:add(nn.Mean(1))

    -- Combined_CNN_RNN_2 = Combined_CNN_RNN_1:clone('weight','bias','gradWeight','gradBias')
    
    local U_mat = nn.Linear(opt.embeddingSize,opt.embeddingSize, false)
    
    local UmulG = nn.Sequential()
    UmulG:add(nn.SelectTable(2))
    UmulG:add(U_mat)
    
    local para_PU = nn.ConcatTable()
    para_PU:add(nn.SelectTable(1))
    para_PU:add(UmulG)
    
    local attention = nn.Sequential()
    attention:add(para_PU)
    attention:add(nn.MM(false,true))
    attention:add(nn.Tanh())
    
    local max_pool1 = nn.Sequential()
    max_pool1:add(nn.Max(2))
    max_pool1:add(nn.SoftMax())
    max_pool1:add(nn.View(1,-1))
    --max_pool1:add(nn.Reshape(opt.sampleSeqLength,1))
    
    local max_pool2 = nn.Sequential()
    max_pool2:add(nn.Max(1))
    max_pool2:add(nn.SoftMax())
    max_pool2:add(nn.View(1,-1))
    --max_pool2:add(nn.Reshape(opt.sampleSeqLength,1))
    
    local pooling = nn.ConcatTable()
    pooling:add(max_pool1)
    pooling:add(max_pool2)
    
    local pool = nn.Sequential()
    pool:add(attention)
    pool:add(pooling)
    
    local mlp2 = nn.ParallelTable()
    mlp2:add(Combined_CNN_LSTM_1)
    mlp2:add(Combined_CNN_LSTM_2)
    mlp2:cuda()
    
    local mlp8 = nn.ConcatTable()
    mlp8:add(nn.Identity())
    mlp8:add(nn.Identity())
    mlp8:add(nn.Identity())
    mlp8:cuda()
    
    local mlp9 = nn.ParallelTable()
    mlp9:add(nn.SelectTable(1))
    mlp9:add(pool)
    mlp9:add(nn.SelectTable(2))
    mlp9:cuda()
    
    local mlp10 = nn.ConcatTable()
    mlp10:add(nn.Identity())
    mlp10:add(nn.Identity())
    mlp10:cuda()
    
    local dot1 = nn.ConcatTable()
    dot1:add(nn.SelectTable(1))
    dot1:add(nn.SelectTable(2))
    
    local atpn1 = nn.Sequential()
    atpn1:add(nn.MM(true,true))
    atpn1:add(nn.Reshape(opt.embeddingSize))
    
    local dot2 = nn.ConcatTable()
    dot2:add(nn.SelectTable(4))
    dot2:add(nn.SelectTable(3))
    
    local atpn2 = nn.Sequential()
    atpn2:add(nn.MM(true,true))
    atpn2:add(nn.Reshape(opt.embeddingSize))
    
    local mlp11 = nn.ParallelTable()
    mlp11:add(dot1)
    mlp11:add(dot2)
    mlp11:cuda()
    
    local mlp12 = nn.ParallelTable()
    mlp12:add(atpn1)
    mlp12:add(atpn2)
    mlp12:cuda()
    
    local mlp3 = nn.ConcatTable()
    mlp3:add(nn.Identity())
    mlp3:add(nn.Identity())
    mlp3:add(nn.Identity())
    mlp3:cuda()

    local mlp4 = nn.ParallelTable()
    mlp4:add(nn.Identity())
    mlp4:add(nn.SelectTable(1))
    mlp4:add(nn.SelectTable(2))
    mlp4:cuda()

    -- used to predict the identity of each person
    local classifierLayer = nn.Linear(opt.embeddingSize,nPersonsTrain)

    -- identification
    local mlp6 = nn.Sequential()
    mlp6:add(classifierLayer)
    mlp6:add(nn.LogSoftMax())
    mlp6:cuda()

    local mlp7 = nn.Sequential()
    mlp7:add(classifierLayer:clone('weight','bias','gradWeight','gradBias'))
    mlp7:add(nn.LogSoftMax())
    mlp7:cuda()

    local mlp5 = nn.ParallelTable()
    mlp5:add(nn.PairwiseDistance(2))
    mlp5:add(mlp6)
    mlp5:add(mlp7)
    mlp5:cuda()

    local fullModel = nn.Sequential()
    fullModel:add(mlp2)
    fullModel:add(mlp8)
    fullModel:add(mlp9)
    fullModel:add(nn.FlattenTable())
    fullModel:add(mlp10)
    fullModel:add(mlp11)
    fullModel:add(mlp12)
    fullModel:cuda()
    
    local finalModel = nn.Sequential()
    finalModel:add(fullModel)
    finalModel:add(mlp3)
    finalModel:add(mlp4)
    finalModel:add(mlp5)
    finalModel:cuda()
    print(finalModel)

    local crit = nn.SuperCriterion()
    crit:add(nn.HingeEmbeddingCriterion(2),1)
    crit:add(nn.ClassNLLCriterion(),1)
    crit:add(nn.ClassNLLCriterion(),1)

    return finalModel, crit, fullModel, cnn
end
