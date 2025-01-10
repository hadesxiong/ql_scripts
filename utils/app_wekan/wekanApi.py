# coding=utf8
import os,json,requests

# 读取环境变量
wekan_url = os.getenv('app_wekan_url')

def userLogin(usr:str,pwd:str) -> dict:

    login_url = f'http://{wekan_url}/users/login'
    login_data = {'username': usr,'password':pwd}
    login_res = requests.post(login_url,data = login_data)
    # print(login_res.json())
    return login_res.json()

def getSwimLanes(token:str,board:str) -> list:

    swimlane_url = f'http://{wekan_url}/api/boards/{board}/swimlanes'
    swimlane_header = { 'Authorization': f'Bearer {token}' }
    swimlane_res = requests.get(swimlane_url, headers=swimlane_header)
    # print(swimlane_res.json())
    return swimlane_res.json()

def getLists(token:str,board:str) -> list:

    list_url = f'http://{wekan_url}/api/boards/{board}/lists'
    list_header = { 'Authorization': f'Bearer {token}' }
    list_res = requests.get(list_url,headers=list_header)
    # print(list_res.json())
    return list_res.json()

def createCard(token:str,board:str,list:str,author:str,title:str,desc:str,swimlane:str) -> dict:

    card_url = f'http://{wekan_url}/api/boards/{board}/lists/{list}/cards'
    card_data = {
        'authorId': author, 'title': title, 'description': desc, 'swimlaneId': swimlane
    }
    card_header = { 'Authorization': f'Bearer {token}' }
    card_res = requests.post(card_url, data=card_data, headers=card_header)
    # print(card_res.json())
    return card_res.json()

def editCard(token:str,board:str,new_swim:str,new_list:str,card:str,body:dict) -> dict:

    card_url = f'http://{wekan_url}/api/boards/{board}/lists/{new_list}/cards/{card}'
    card_header = { 'Authorization': f'Bearer {token}' }
    body.update({
        'newBoardId': board,
        'newSwimlaneId': new_swim,
        'newListId': new_list
    })

    card_res = requests.put(card_url, data=body, headers=card_header)
    # print(card_res.json())
    return card_res.json()

def getCheckLists(token:str,board:str,card:str):

    checklist_url = f'http://{wekan_url}/api/boards/{board}/cards/{card}/chekclists'
    checklist_header = { 'Authorization': f'Bearer {token}' }

    checklist_res = requests.get(checklist_url, headers=checklist_header)
    # print(checklist_res.json())
    return checklist_res.json()

def createCheckList(token:str,board:str,card:str, body:dict) -> dict:

    checklist_url = f'http://{wekan_url}/api/boards/{board}/cards/{card}/checklists'
    checklist_header = { 'Authorization': f'Bearer {token}' }

    checklist_res = requests.post(checklist_url, data=body,headers=checklist_header)
    # print(checklist_res.json())
    return checklist_res.json()
    