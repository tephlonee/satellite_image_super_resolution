data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "ec2_role" {
  name               = "${var.project}-ec2-runner"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json

  tags = {
    Project = var.project
  }
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

resource "aws_s3_bucket" "artifacts" {
  count         = var.artifacts_bucket == "" ? 1 : 0
  bucket_prefix = var.artifacts_bucket_prefix

  tags = {
    Project = var.project
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  count  = var.artifacts_bucket == "" ? 1 : 0
  bucket = aws_s3_bucket.artifacts[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  count  = var.artifacts_bucket == "" ? 1 : 0
  bucket = aws_s3_bucket.artifacts[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  count  = var.artifacts_bucket == "" ? 1 : 0
  bucket = aws_s3_bucket.artifacts[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

locals {
  resolved_artifacts_bucket = var.artifacts_bucket != "" ? var.artifacts_bucket : aws_s3_bucket.artifacts[0].bucket
}

data "aws_iam_policy_document" "s3_access" {
  statement {
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = ["arn:aws:s3:::${local.resolved_artifacts_bucket}"]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${var.runs_prefix}/*"]
    }
  }

  statement {
    effect  = "Allow"
    actions = ["s3:PutObject", "s3:AbortMultipartUpload", "s3:GetObject", "s3:GetObjectVersion"]
    resources = [
      "arn:aws:s3:::${local.resolved_artifacts_bucket}/${var.runs_prefix}/*"
    ]
  }

  dynamic "statement" {
    for_each = var.dataset_bucket != "" ? [1] : []
    content {
      effect  = "Allow"
      actions = ["s3:ListBucket"]
      resources = [
        "arn:aws:s3:::${var.dataset_bucket}"
      ]
      dynamic "condition" {
        for_each = var.dataset_prefix != "" ? [1] : []
        content {
          test     = "StringLike"
          variable = "s3:prefix"
          values   = ["${var.dataset_prefix}*"]
        }
      }
    }
  }

  dynamic "statement" {
    for_each = var.dataset_bucket != "" ? [1] : []
    content {
      effect  = "Allow"
      actions = ["s3:GetObject", "s3:GetObjectVersion"]
      resources = [
        var.dataset_prefix != "" ? "arn:aws:s3:::${var.dataset_bucket}/${var.dataset_prefix}*" : "arn:aws:s3:::${var.dataset_bucket}/*"
      ]
    }
  }
}

resource "aws_iam_policy" "s3_access" {
  name   = "${var.project}-ec2-s3-access"
  policy = data.aws_iam_policy_document.s3_access.json
}

resource "aws_iam_role_policy_attachment" "s3_access_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.s3_access.arn
}
