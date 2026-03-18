import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tmf_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation



# wyh

def square_distance(src, dst):
    """
    Calculate squared Euclidean distance between each two points.

    Args:
        src: [B, N, C]
        dst: [B, M, C]

    Returns:
        dist: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Index points according to idx.

    Args:
        points: [B, N, C]
        idx: [B, S] or [B, S, K]

    Returns:
        new_points: [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(k, xyz, new_xyz):
    """
    KNN search.

    Args:
        k: number of neighbors
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]

    Returns:
        idx: [B, S, k]
    """
    dist = square_distance(new_xyz, xyz)
    idx = dist.topk(k=k, dim=-1, largest=False, sorted=False)[1]
    return idx



class InputTransformNet(nn.Module):
    """
    Lightweight T-Net for xyz alignment.
    Input:  [B, 3, N]
    Output: [B, 3, 3]
    """
    def __init__(self):
        super(InputTransformNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, xyz):
        B = xyz.size(0)

        x = F.relu(self.bn1(self.conv1(xyz)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(3, device=xyz.device).view(1, 9).repeat(B, 1)
        x = x + identity
        x = x.view(B, 3, 3)
        return x



# EFM
class EnhancingFeatureModule(nn.Module):

    def __init__(self, in_channels=9, topo_out_channels=64, k=16, transformer_heads=4):
        super(EnhancingFeatureModule, self).__init__()
        self.in_channels = in_channels
        self.topo_out_channels = topo_out_channels
        self.k = k

        self.input_tnet = InputTransformNet()


        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)


        edge_in_channels = (64 + 128 + 256 + 512) * 2 + 3  # 1923


        self.edge_conv1 = nn.Conv2d(edge_in_channels, 1024, 1)
        self.edge_conv2 = nn.Conv2d(1024, 128, 1)
        self.edge_conv3 = nn.Conv2d(128, topo_out_channels, 1)

        self.edge_bn1 = nn.BatchNorm2d(1024)
        self.edge_bn2 = nn.BatchNorm2d(128)
        self.edge_bn3 = nn.BatchNorm2d(topo_out_channels)


        self.edge_score = nn.Sequential(
            nn.Conv2d(edge_in_channels, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )


        self.node_gate = nn.Sequential(
            nn.Conv1d(topo_out_channels, topo_out_channels, 1),
            nn.BatchNorm1d(topo_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(topo_out_channels, topo_out_channels, 1),
            nn.Sigmoid()
        )


        self.bmm_proj = nn.Sequential(
            nn.Conv1d(topo_out_channels, topo_out_channels, 1),
            nn.BatchNorm1d(topo_out_channels),
            nn.ReLU(inplace=True)
        )

    def _extract_multilevel_features(self, x):
        """
        x: [B, C, N]
        return: [B, 960, N]
        """
        f1 = F.relu(self.bn1(self.conv1(x)))   # [B, 64, N]
        f2 = F.relu(self.bn2(self.conv2(f1)))  # [B,128, N]
        f3 = F.relu(self.bn3(self.conv3(f2)))  # [B,256, N]
        f4 = F.relu(self.bn4(self.conv4(f3)))  # [B,512, N]
        feat = torch.cat([f1, f2, f3, f4], dim=1)  # [B,960,N]
        return feat

    def forward(self, x):
        B, C, N = x.shape
        xyz = x[:, :3, :]  # [B,3,N]


        trans = self.input_tnet(xyz)                 
        xyz_trans = torch.bmm(trans, xyz)            


        x_trans = torch.cat([xyz_trans, x[:, 3:, :]], dim=1) if C > 3 else xyz_trans

        feat_orig = self._extract_multilevel_features(x)        
        feat_trans = self._extract_multilevel_features(x_trans) 

        xyz_t = xyz_trans.transpose(1, 2).contiguous()          
        feat_orig_t = feat_orig.transpose(1, 2).contiguous()    
        feat_trans_t = feat_trans.transpose(1, 2).contiguous()  


        idx = knn_point(self.k, xyz_t, xyz_t)              

        neighbor_xyz = index_points(xyz_t, idx)              
        center_xyz = xyz_t.unsqueeze(2).repeat(1, 1, self.k, 1)
        relative_xyz = neighbor_xyz - center_xyz           

        neighbor_feat = index_points(feat_trans_t, idx)         
        center_feat = feat_orig_t.unsqueeze(2).repeat(1, 1, self.k, 1)

   
        edge_feat = torch.cat([center_feat, neighbor_feat, relative_xyz], dim=-1)   
        edge_feat = edge_feat.permute(0, 3, 1, 2).contiguous()                     

        local_feat = F.relu(self.edge_bn1(self.edge_conv1(edge_feat)))
        local_feat = F.relu(self.edge_bn2(self.edge_conv2(local_feat)))
        local_feat = F.relu(self.edge_bn3(self.edge_conv3(local_feat)))        


        edge_logits = self.edge_score(edge_feat)                           
        edge_weight = F.softmax(edge_logits, dim=-1)
        local_feat = torch.sum(local_feat * edge_weight, dim=-1)               


        gate = self.node_gate(local_feat)                                      
        local_feat = local_feat * gate


        affinity = torch.bmm(xyz_t, xyz_t.transpose(1, 2))                  
        affinity = F.softmax(affinity, dim=-1)

        topo_feat = torch.bmm(local_feat, affinity)                      
        topo_feat = self.bmm_proj(topo_feat)                    

        return xyz_trans, topo_feat, trans



# ATM

class AttentionTopologyModule(nn.Module):
    """
    Topology-aware attention module.

    Input:
        xyz:   [B, 3, N]
        feats: [B, C, N]
    Output:
        out:   [B, C, N]
    """
    def __init__(self, channels, k=16, hidden_dim=64):
        super(AttentionTopologyModule, self).__init__()
        self.channels = channels
        self.k = k

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(channels * 2 + 3, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1)
        )

        self.value_mlp = nn.Sequential(
            nn.Conv2d(channels + 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz, feats):
        B, C, N = feats.shape

        xyz_t = xyz.transpose(1, 2).contiguous()   
        feats_t = feats.transpose(1, 2).contiguous() 

        idx = knn_point(self.k, xyz_t, xyz_t)       

        neighbor_xyz = index_points(xyz_t, idx)     
        center_xyz = xyz_t.unsqueeze(2).repeat(1, 1, self.k, 1)
        relative_xyz = neighbor_xyz - center_xyz

        neighbor_feat = index_points(feats_t, idx)   
        center_feat = feats_t.unsqueeze(2).repeat(1, 1, self.k, 1)

        attn_input = torch.cat([center_feat, neighbor_feat, relative_xyz], dim=-1)  
        attn_input = attn_input.permute(0, 3, 1, 2).contiguous()                    
        attn_logits = self.attn_mlp(attn_input)                                    
        attn = F.softmax(attn_logits, dim=-1)

        value_input = torch.cat([neighbor_feat, relative_xyz], dim=-1)      
        value_input = value_input.permute(0, 3, 1, 2).contiguous()            
        values = self.value_mlp(value_input)                            

        out = torch.sum(attn * values, dim=-1)                            
        out = self.out_conv(out) + feats
        return out



# 3-Conv

class RGB3ConvExtractor(nn.Module):
    """
    3-Conv RGB feature extractor.
    Input:  [B, 3, N]
    Output: [B, out_channels, N]
    """
    def __init__(self, out_channels=128):
        super(RGB3ConvExtractor, self).__init__()

        self.conv1 = nn.Conv1d(3, 16, 1)
        self.conv2 = nn.Conv1d(3, 32, 1)
        self.conv3 = nn.Conv1d(3, 64, 1)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)

        self.fuse = nn.Sequential(
            nn.Conv1d(16 + 32 + 64, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb):
        f1 = F.relu(self.bn1(self.conv1(rgb)))  
        f2 = F.relu(self.bn2(self.conv2(rgb))) 
        f3 = F.relu(self.bn3(self.conv3(rgb)))

        feat = torch.cat([f1, f2, f3], dim=1)   
        feat = self.fuse(feat)          
        return feat



# TMF model

class get_model(nn.Module):
    def __init__(self, num_classes, input_channels=9):
        super(get_model, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        self.efm = EnhancingFeatureModule(
            in_channels=input_channels,
            topo_out_channels=64,
            k=16
        )

        efm_out_channels = 64

        self.l0_skip_conv = nn.Sequential(
            nn.Conv1d(efm_out_channels, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.sa1 = PointNetSetAbstractionMsg(
            1024, [0.05, 0.1], [16, 32],
            efm_out_channels,
            [[16, 16, 32], [32, 32, 64]]
        )   # 96

        self.sa2 = PointNetSetAbstractionMsg(
            256, [0.1, 0.2], [16, 32],
            96,
            [[64, 64, 128], [64, 96, 128]]
        )   # 256

        self.sa3 = PointNetSetAbstractionMsg(
            64, [0.2, 0.4], [16, 32],
            256,
            [[128, 196, 256], [128, 196, 256]]
        )   # 512

        self.sa4 = PointNetSetAbstractionMsg(
            16, [0.4, 0.8], [16, 32],
            512,
            [[256, 256, 512], [256, 384, 512]]
        )   # 1024

        self.fp4 = PointNetFeaturePropagation(1024 + 512, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 96, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + 32, [128, 128, 128])

        self.atm_l3 = AttentionTopologyModule(channels=256, k=16, hidden_dim=64)
        self.atm_l2 = AttentionTopologyModule(channels=256, k=8, hidden_dim=64)
        self.atm_l1 = AttentionTopologyModule(channels=128, k=8, hidden_dim=32)
        # self.atm_l0 = AttentionTopologyModule(channels=128, k=4, hidden_dim=32)

        self.rgb_extractor = RGB3ConvExtractor(out_channels=128)

        self.fusion_conv = nn.Sequential(
            nn.Conv1d(128 + 128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):

        B, C, N = xyz.shape
        assert C >= 6, "Input channel."

        if C < self.input_channels:
            pad = torch.zeros(B, self.input_channels - C, N, device=xyz.device, dtype=xyz.dtype)
            x_in = torch.cat([xyz, pad], dim=1)
        else:
            x_in = xyz[:, :self.input_channels, :]

        l0_rgb = x_in[:, 3:6, :]

        l0_xyz, l0_points, trans_feat = self.efm(x_in)
        l0_skip = self.l0_skip_conv(l0_points)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l3_points = self.atm_l3(l3_xyz, l3_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points = self.atm_l2(l2_xyz, l2_points)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points = self.atm_l1(l1_xyz, l1_points)

        l0_points = self.fp1(l0_xyz, l1_xyz, l0_skip, l1_points)
        # l0_points = self.atm_l0(l0_xyz, l0_points)

        rgb_feat = self.rgb_extractor(l0_rgb)

        fused_feat = torch.cat([l0_points, rgb_feat], dim=1)
        fused_feat = self.fusion_conv(fused_feat)

        logits = self.classifier(fused_feat)
        logits = F.log_softmax(logits, dim=1)
        logits = logits.permute(0, 2, 1)

        return logits, trans_feat



class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat=None, weight=None):
        """
        Args:
            pred: [B, N, num_classes]
            target: [B, N]
            trans_feat: [B, 3, 3] or None
            weight: class weights or None
        """
        pred = pred.contiguous().view(-1, pred.shape[-1])
        target = target.view(-1)

        cls_loss = F.nll_loss(pred, target, weight=weight)

        if trans_feat is not None:
            B = trans_feat.size(0)
            I = torch.eye(3, device=trans_feat.device).unsqueeze(0).repeat(B, 1, 1)
            mat_diff = torch.bmm(trans_feat, trans_feat.transpose(2, 1)) - I
            mat_diff_loss = torch.mean(torch.norm(mat_diff, dim=(1, 2)))
            total_loss = cls_loss + self.mat_diff_loss_scale * mat_diff_loss
        else:
            total_loss = cls_loss

        return total_loss



if __name__ == '__main__':
    model = get_model(num_classes=20, input_channels=9)
    xyz = torch.rand(2, 9, 2048) 
    pred, trans = model(xyz)

    print('pred shape:', pred.shape) 
    print('trans shape:', trans.shape) 

    target = torch.randint(0, 20, (2, 2048))
    criterion = get_loss()
    loss = criterion(pred, target, trans_feat=trans)
    print('loss:', loss.item())
