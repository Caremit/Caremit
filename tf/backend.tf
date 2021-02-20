terraform {
  backend "azurerm" {
    resource_group_name   = "caremit-rg"
    storage_account_name  = "caremitws1805579988"
    container_name        = "terraform"
    key                   = "tfstate"
  }
}
