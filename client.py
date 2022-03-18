import argparse
import asyncio
import socketio
import random
import signal
import sys
from hacktrick_agent import HacktrickAgent


sio = socketio.AsyncClient()

settings = {}
agent = HacktrickAgent()

@sio.event
async def connect():
    print('connection established')


@sio.event
async def start_game(data):
    print('start_game received with ', data)
    # await sio.emit('my response', {'response': 'my response'})

@sio.event
async def end_game(data):
    print('end_game received with ', data)
    # await sio.emit('my response', {'response': 'my response'})

@sio.event
async def state_pong(data):
    action =  agent.action(data)
    
    if "collaborative" in settings['mode']:
        print("actions", action)
        await sio.emit('action_collaborative', {'actions':action,'team_name': settings['team_name']})

    else:
        print("action", action)        
        await sio.emit('action', {'action': action})
    
    score = data['state']['score']
    state = data['state']['state']
    print("score:", score)
        

@sio.event
async def end_game(data):
    print('end_game received with ', data)
    # await sio.emit('my response', {'response': 'my response'})
    await sio.disconnect()

@sio.event
async def waiting(data):
    print('waiting received with ', data)
   
@sio.event
async def creation_failed(data):
    print('Failed to create game')
    print('Received the following error', data['error'])  

@sio.event
async def reset_game(data):
    print('creation_failed received with ', data)

@sio.event
async def disconnect():
    print('disconnected from server')

@sio.event
async def authentication_error(data):
    print('authentication_error received')


async def main():
    await sio.connect('http://ec2-3-14-245-107.us-east-2.compute.amazonaws.com/') ## Change here to aws url
    await sio.emit('create', {'mode': settings['mode'],'team_name': settings['team_name'], 'password':settings['password'], 'layout':settings['layout']})
    await sio.wait()


async def signal_handler(signal, frame):
     print ('You pressed Ctrl+C - or killed me with -2')
     #.... Put your logic here .....
     await sio.disconnect()
     sys.exit(0)

if __name__ == '__main__':
    modes = ["single" ,"collaborative"]
    layouts = [
        "leaderboard_single",
        "leaderboard_collaborative",
        "round_of_16_single",
        "round_of_16_collaborative",
        "quarter_final_single",
        "quarter_final_collaborative",
        "semi_final_single",
        "semi_final_collaborative",
        "final_single",
        "final_collaborative"
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('--team_name', type=str, required=True)
    parser.add_argument('--password', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--layout', type=str, required=True)
    args = parser.parse_args()

    if args.mode not in modes or \
        args.layout not in layouts:
        print("invalid parameters have been entered. Please ensure mode and layout are correct")
        sys.exit(0)
    settings['team_name'] = args.team_name
    settings['password'] = args.password
    settings['mode'] = args.mode
    settings['layout'] = args.layout

    print(settings)
    # signal.signal(signal.SIGINT, signal_handler)
    agent.set_mode(settings['mode'])
    asyncio.run(main())
    # asyncio.get_event_loop().run_until_complete(main(args.host, args.team_name, args.password))