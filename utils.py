import torch
from torch.optim import Adam


def train(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername=""):
    optimizer = Adam(model.parameters(), lr=config["lr"])


    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1500, gamma=.1
    )
    print(lr_scheduler.get_last_lr())
    best_valid_loss = 1e10

    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        count = 0
        model.train()

        for batch_no, (clean_batch, noisy_batch, label_batch) in enumerate(train_loader, start=1):
            clean_batch, noisy_batch, label_batch = clean_batch.to(device), noisy_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()

            loss = model(clean_batch, noisy_batch, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            avg_loss += loss.item()
            count += 1
            # Display progress every 100 batches
            if batch_no % 100 == 0:
                print(f"    🔄 Batch {batch_no}: Training Loss = {loss.item():.6f}")

        avg_loss /= count
        print(f"🔹 Epoch {epoch_no + 1} - Average Training Loss: {avg_loss:.6f}")

        lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():

                for batch_no, (clean_batch, noisy_batch, label_batch) in enumerate(valid_loader, start=1):
                    clean_batch, noisy_batch, label_batch = clean_batch.to(device), noisy_batch.to(device), label_batch.to(device)
                    loss = model(clean_batch, noisy_batch, label_batch)
                    avg_loss_valid += loss.item()

            avg_valid_loss = avg_loss_valid / batch_no
            print(f"🟡 Validation Loss: {avg_valid_loss:.6f}")

            if best_valid_loss > avg_loss_valid / batch_no:
                best_valid_loss = avg_loss_valid / batch_no
                print("\n best loss is updated to ", avg_loss_valid / batch_no, "at", epoch_no, )

                if foldername != "":
                    torch.save(model.state_dict(), output_path)

    torch.save(model.state_dict(), final_path)