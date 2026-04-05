data "aws_ssm_parameter" "dlami_gpu_ami" {
  count = var.ami_id == null ? 1 : 0
  name  = var.dlami_ssm_parameter_name
}

data "aws_subnet" "selected" {
  id = local.resolved_subnet_id
}

locals {
  resolved_ami = var.ami_id != null ? var.ami_id : data.aws_ssm_parameter.dlami_gpu_ami[0].value
}

resource "aws_ebs_volume" "data" {
  availability_zone = data.aws_subnet.selected.availability_zone
  size              = var.data_volume_size_gb
  type              = "gp3"

  tags = {
    Name    = "${var.project}-data"
    Project = var.project
  }
}

resource "aws_volume_attachment" "data" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.data.id
  instance_id = aws_instance.trainer.id
}

resource "aws_instance" "trainer" {
  ami                         = local.resolved_ami
  instance_type               = var.instance_type
  subnet_id                   = local.resolved_subnet_id
  vpc_security_group_ids      = [aws_security_group.ec2_sg.id]
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name
  associate_public_ip_address = true
  key_name                    = var.key_name

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
        max_price                      = var.spot_max_price != "" ? var.spot_max_price : null
      }
    }
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    region             = var.region
    github_repo_url    = var.github_repo_url
    github_ref         = var.github_ref
    artifacts_bucket   = local.resolved_artifacts_bucket
    runs_prefix        = var.runs_prefix
    data_root          = var.data_root
    data_subdir        = var.data_subdir
    dataset_s3_uri     = var.dataset_s3_uri
    s3_no_sign_request = var.s3_no_sign_request ? "1" : "0"
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = 100
  }

  tags = {
    Name    = "${var.project}-trainer"
    Project = var.project
    Owner   = "tijani"
    Env     = "dev"
  }
}
