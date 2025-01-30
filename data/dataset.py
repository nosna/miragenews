import torch

class MiRAGeImageOrTextDataset(torch.utils.data.Dataset):
    def __init__(self, real_pt, fake_pt):
        # Load real and fake tensors
        real_pt = torch.load(real_pt)
        fake_pt = torch.load(fake_pt)
        
        # Concatenate real and fake images
        self.all_data = torch.cat((real_pt, fake_pt), dim=0)
        self.labels = torch.cat((torch.zeros(len(real_pt)), torch.ones(len(fake_pt))))
    
    def __getitem__(self, index):
        features = self.all_data[index]
        label = self.labels[index].item()
        return features, label

    def __len__(self):
        return len(self.all_data)
    

class MiRAGeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, real_img_pt, fake_img_pt, real_text_pt, fake_text_pt):
        # Load real and fake tensors
        real_img_pt = torch.load(real_img_pt)
        fake_img_pt = torch.load(fake_img_pt)
        real_text_pt = torch.load(real_text_pt)
        fake_text_pt = torch.load(fake_text_pt)
        
        # Concatenate real and fake news
        self.all_imgs = torch.cat((real_img_pt, fake_img_pt), dim=0)
        self.all_texts = torch.cat((real_text_pt, fake_text_pt), dim=0)
        self.labels = torch.cat((torch.zeros(len(real_img_pt)), torch.ones(len(fake_img_pt))))
    
    def __getitem__(self, index):
        image_features = self.all_imgs[index]
        text_features = self.all_texts[index]
        label = self.labels[index].item()
        return image_features, text_features, label

    def __len__(self):
        return len(self.all_imgs)
